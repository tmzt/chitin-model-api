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
// JS → Rust → UDS path:
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

  const resp = await c.inference({
    role: 'deep',
    input: 'hello from node',
    maxTokens: 64,
    sessionId: 's-node-test',
  });
  console.log('inference response:');
  console.log(`  text: ${JSON.stringify(resp.text)}`);
  console.log(`  sessionId: ${resp.sessionId}`);
  console.log(`  rawText: ${JSON.stringify(resp.rawText)}`);
  console.log(`  injections: ${resp.injections.length}`);

  await c.shutdown();
  console.log('shutdown complete');
})().catch((e) => {
  console.error('FAIL:', e);
  process.exit(1);
});
