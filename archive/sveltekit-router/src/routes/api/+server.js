export async function GET() {
    const data = {
      message: 'Hello from SvelteKit API!',
    };
    return new Response(JSON.stringify(data), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }
  
  export async function POST({ request }) {
    const body = await request.json();
    const data = {
      message: `Received data: ${JSON.stringify(body)}`,
    };
    return new Response(JSON.stringify(data), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }