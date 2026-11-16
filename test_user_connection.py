
from db import get_db_connection

connection = get_db_connection()

if connection:
    try:
        cursor = connection.cursor()

        # Insert a test user
        cursor.execute(
            """
            INSERT INTO test_users (name, email)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE name = VALUES(name)
            """,
            ("Test User", "testuser@example.com")
        )
        connection.commit()
        print("Test user inserted successfully!")

        # Retrieve the test user
        cursor.execute(
            "SELECT id, name, email FROM test_users WHERE email = %s",
            ("testuser@example.com",)
        )

        user = cursor.fetchone()

        if user:
            print("Retrieved test user:", user)

        cursor.close()

    except Exception as error:
        connection.rollback()
        print("Error:", error)

    finally:
        connection.close()
else:
    print("Could not connect to MySQL.")