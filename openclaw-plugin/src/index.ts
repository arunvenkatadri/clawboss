import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

interface BridgeTool {
  name: string;
  description: string;
  parameters: {
    type: "object";
    properties: Record<
      string,
      { type: string; description?: string; default?: unknown }
    >;
    required?: string[];
  };
}

interface BridgeResponse {
  success: boolean;
  result: unknown;
  error: { kind: string; message: string } | null;
  metadata: {
    duration_ms: number;
    tool_name: string;
    budget?: {
      tokens_used: number;
      token_limit: number | null;
      iterations: number;
      iteration_limit: number;
    };
  };
}

export default definePluginEntry({
  id: "clawboss-bridge",
  name: "Clawboss Supervised Tools",

  async register(api) {
    const config = api.getConfig();
    const bridgeUrl =
      (config.bridgeUrl as string) || "http://localhost:9229";

    // Fetch available tools from the bridge
    let tools: BridgeTool[];
    try {
      const resp = await fetch(`${bridgeUrl}/tools`);
      if (!resp.ok) {
        api.log.error(
          `Failed to fetch tools from clawboss bridge at ${bridgeUrl}: HTTP ${resp.status}`
        );
        return;
      }
      const data = (await resp.json()) as { tools: BridgeTool[] };
      tools = data.tools;
    } catch (err) {
      api.log.error(
        `Cannot connect to clawboss bridge at ${bridgeUrl}. Is the bridge running?`
      );
      return;
    }

    // Register each tool with OpenClaw
    for (const tool of tools) {
      api.registerTool({
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,

        async execute(
          _callId: string,
          params: Record<string, unknown>
        ) {
          const resp = await fetch(`${bridgeUrl}/execute/${tool.name}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ params }),
          });

          if (!resp.ok) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Bridge returned HTTP ${resp.status}`,
                },
              ],
              isError: true,
            };
          }

          const data = (await resp.json()) as BridgeResponse;

          if (data.success) {
            const text =
              typeof data.result === "string"
                ? data.result
                : JSON.stringify(data.result);
            return {
              content: [{ type: "text" as const, text }],
            };
          } else {
            return {
              content: [
                {
                  type: "text" as const,
                  text:
                    data.error?.message ||
                    "Unknown error from clawboss bridge",
                },
              ],
              isError: true,
            };
          }
        },
      });
    }

    api.log.info(
      `Registered ${tools.length} tool(s) from clawboss bridge`
    );
  },
});
