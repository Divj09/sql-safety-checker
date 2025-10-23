import random
import csv
import os

BENIGN = [
    "SELECT * FROM users WHERE id = 1;",
    "UPDATE products SET price = 100 WHERE id = 5;",
    "INSERT INTO orders (user_id, amount) VALUES (10, 250);",
    "DELETE FROM logs WHERE date < '2024-01-01';",
    "SELECT name, email FROM customers WHERE email = 'test@example.com';"
]

ATTACK_SNIPPETS = [
    "' OR '1'='1'; --",
    "' OR 1=1 --",
    "'; DROP TABLE users; --",
    "' UNION SELECT username, password FROM admin --",
    "admin'--",
    "'; EXEC xp_cmdshell('dir'); --"
]

def generate(n_pairs=300, out="data/queries_labeled.csv"):
    os.makedirs("data", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_text", "label"])
        for _ in range(n_pairs):
            b = random.choice(BENIGN)
            m = b + " " + random.choice(ATTACK_SNIPPETS)
            writer.writerow([b, 0])
            writer.writerow([m, 1])
    print("Saved", out)

if __name__ == "__main__":
    generate(300)
