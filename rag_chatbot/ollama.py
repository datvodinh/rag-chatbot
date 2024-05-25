import asyncio
import threading
import socket


def run_ollama_server():
    async def run_process(cmd):
        print('>>> starting', *cmd)
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # env={**os.environ, 'OLLAMA_NUM_PARALLEL': '8', 'OLLAMA_MAX_LOADED_MODELS': '1'}
        )

        # define an async pipe function
        async def pipe(lines):
            async for line in lines:
                print(line.decode().strip())

            await asyncio.gather(
                pipe(process.stdout),
                pipe(process.stderr),
            )

        # call it
        await asyncio.gather(pipe(process.stdout), pipe(process.stderr))

    async def start_ollama_serve():
        await run_process(['ollama', 'serve'])

    def run_async_in_thread(loop, coro):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(coro)
        loop.close()

    # Create a new event loop that will run in a new thread
    new_loop = asyncio.new_event_loop()

    # Start ollama serve in a separate thread so the cell won't block execution
    thread = threading.Thread(target=run_async_in_thread, args=(new_loop, start_ollama_serve()))
    thread.start()


def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(('localhost', port))
            return True
        except ConnectionRefusedError:
            return False
