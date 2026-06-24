// Smoke test for the Node bindings. Assumes a `chitin-model-api`
// server is already running on the socket path passed as argv[2]
// (or /tmp/chitin-model-api.sock by default).
//
// Run with one of:
//   node test.js                                  (default socket)
//   node test.js /tmp/my-custom.sock
//
// The StubSlot-backed binary (built without --features llama-cpp)
// echoes prompts back, which is enough to exercise the full
// JS → Rust → UDS path. With tools in the request, the stub also
// emits a canned ToolCall — useful for verifying the agent loop
// shape end-to-end without a real model:
//
//   cargo build -p model_api_server --bin chitin-model-api --no-default-features
//   /tmp/cm-target/debug/chitin-model-api --socket /tmp/test.sock &
//   node test.js /tmp/test.sock

const { Client } = require('./index.js');

const SOCKET = process.argv[2] || '/tmp/chitin-model-api.sock';

(async () => {
  console.log(`connecting to ${SOCKET}…`);
  const c = await Client.connect(SOCKET);
  console.log(`connected: model=${c.modelName} gpuMb=${c.gpuMemoryMb} proto=${c.protocolVersion}`);

  // ── Plain chat ────────────────────────────────────────────────────
  console.log('\n[1] plain inference (no tools):');
  const r1 = await c.inference({
    role: 'deep',
    input: 'hello from node',
    maxTokens: 64,
    sessionId: 's-node-test-plain',
    temperature: 0.7,
  });
  console.log(`  text:      ${JSON.stringify(r1.text)}`);
  console.log(`  sessionId: ${r1.sessionId}`);
  console.log(`  toolCalls: ${r1.toolCalls.length}`);

  // ── Tool round-trip (agent loop turn 1) ──────────────────────────
  console.log('\n[2] inference with a tool (server returns toolCalls):');
  const r2 = await c.inference({
    role: 'deep',
    input: 'what is 2+2?',
    maxTokens: 64,
    sessionId: 's-node-test-tools',
    toolMode: 'client',           // make the server bubble tool calls back
    tools: [{
      name: 'calculator',
      description: 'Evaluate a math expression',
      parameters: [
        { name: 'expr', type: 'string', required: true, description: 'The expression' },
      ],
    }],
  });
  console.log(`  text:      ${JSON.stringify(r2.text)}`);
  console.log(`  toolCalls: ${r2.toolCalls.length}`);
  for (const tc of r2.toolCalls) {
    console.log(`    - id=${tc.id} name=${tc.name} args=${tc.argumentsJson}`);
  }

  // ── Agent loop turn 2: send result back ──────────────────────────
  if (r2.toolCalls.length > 0) {
    console.log('\n[3] sending tool result back:');
    const tc = r2.toolCalls[0];
    const r3 = await c.inference({
      role: 'deep',
      input: '',                  // model has the prior context via session
      maxTokens: 64,
      sessionId: 's-node-test-tools',
      toolResults: [{ callId: tc.id, output: '4' }],
    });
    console.log(`  text: ${JSON.stringify(r3.text)}`);
  }

  await c.shutdown();
  console.log('\nshutdown complete');
})().catch((e) => {
  console.error('FAIL:', e);
  process.exit(1);
});
