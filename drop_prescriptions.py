import aiomysql
import asyncio
import os
from dotenv import load_dotenv

load_dotenv('.env')

async def drop_table():
    pool = await aiomysql.create_pool(
        host=os.getenv('MYSQL_HOST'),
        port=int(os.getenv('MYSQL_PORT')),
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        db=os.getenv('MYSQL_DB'),
        autocommit=True
    )
    
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute('DROP TABLE IF EXISTS prescriptions')
            print('✅ Prescriptions table dropped successfully')
    
    pool.close()
    await pool.wait_closed()

asyncio.run(drop_table())
