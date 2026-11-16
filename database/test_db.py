"""
test_db.py — Phase 1 database test.

Tests:
  1. Connection works with current DB_DRIVER setting.
  2. INSERT into Image succeeds.
  3. INSERT into Feedback with valid rating (1–5) succeeds.
  4. INSERT into Feedback with rating=6  → DB rejects it (CHECK constraint).
  5. FK violation: Feedback referencing a non-existent image_id is rejected.

Run:
    cd database
    python test_db.py
"""

import os, sys, traceback

# ── Load .env manually (no dotenv library needed) ─────────────────────────────
env_path = os.path.join(os.path.dirname(__file__), "..", "backend", ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, os.path.dirname(__file__))
from db import get_connection

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def run(label, fn):
    try:
        fn()
        print(f"{PASS}  {label}")
        return True
    except Exception as exc:
        print(f"{FAIL}  {label}")
        print(f"       → {exc}")
        return False


def test_all():
    conn = get_connection()
    cur  = conn.cursor()
    driver = os.environ.get("DB_DRIVER", "mssql").lower()

    # Helper: driver-agnostic placeholder
    ph = "?" if driver == "mssql" else "%s"

    inserted_id = None

    # ── Test 1: Insert a valid image row ──────────────────────────────────────
    def t1():
        nonlocal inserted_id
        sql = (
            f"INSERT INTO Image (original_image, enhanced_image, processing_info) "
            f"VALUES ({ph}, {ph}, {ph})"
        )
        cur.execute(sql, ("storage/test_orig.jpg", None, '{"patches":4,"inference_ms":200}'))
        conn.commit()
        # Retrieve the auto-generated PK
        if driver == "mssql":
            cur.execute("SELECT @@IDENTITY")
        else:
            cur.execute("SELECT LAST_INSERT_ID()")
        inserted_id = int(cur.fetchone()[0])

    run("Insert valid Image row", t1)

    # ── Test 2: Insert valid Feedback (rating=4) ──────────────────────────────
    def t2():
        cur.execute(
            f"INSERT INTO Feedback (image_id, rating) VALUES ({ph}, {ph})",
            (inserted_id, 4)
        )
        conn.commit()

    run("Insert Feedback with rating=4 (valid)", t2)

    # ── Test 3: Rating=6 must be rejected by CHECK constraint ─────────────────
    def t3():
        try:
            cur.execute(
                f"INSERT INTO Feedback (image_id, rating) VALUES ({ph}, {ph})",
                (inserted_id, 6)
            )
            conn.commit()
            raise AssertionError("DB accepted rating=6 — CHECK constraint missing!")
        except AssertionError:
            raise
        except Exception:
            conn.rollback()   # expected rejection — test passes

    run("DB rejects rating=6 (CHECK constraint)", t3)

    # ── Test 4: FK violation — non-existent image_id ─────────────────────────
    def t4():
        try:
            cur.execute(
                f"INSERT INTO Feedback (image_id, rating) VALUES ({ph}, {ph})",
                (999999, 3)
            )
            conn.commit()
            raise AssertionError("DB accepted orphan FK — FK constraint missing!")
        except AssertionError:
            raise
        except Exception:
            conn.rollback()   # expected rejection — test passes

    run("DB rejects orphan image_id FK violation", t4)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if inserted_id:
        cur.execute(f"DELETE FROM Image WHERE image_id = {ph}", (inserted_id,))
        conn.commit()

    cur.close()
    conn.close()
    print("\nPhase 1 database tests complete.")


if __name__ == "__main__":
    test_all()
