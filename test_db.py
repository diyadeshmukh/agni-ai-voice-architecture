from dotenv import load_dotenv
import os
import psycopg2

load_dotenv(override=True)

print("DB_PASSWORD loaded:", bool(os.getenv("DB_PASSWORD")))
print("DB_PASSWORD length:", len(os.getenv("DB_PASSWORD", "")))

password = input("Enter the working PostgreSQL password: ")

print("ENV and typed password match:", password == os.getenv("DB_PASSWORD"))

try:
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=5432,
        dbname="agni_ai_dev",
        user="postgres",
        password=password
    )

    print("PYTHON DATABASE CONNECTION SUCCESS")
    conn.close()

except Exception as e:
    print("PYTHON DATABASE CONNECTION FAILED")
    print(e)