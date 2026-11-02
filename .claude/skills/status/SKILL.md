---
name: status
description: Status of every strategy (service, HOD/ORB paper books, MOM sleeve, rails, crons) — run when the owner asks how the strategies are doing
---

Run `python3 scripts/book_status.py` from /home/ec2-user/onemil (read-only; takes ~15-20 s because it greps today's
journal). Relay the output verbatim to the owner, then add ONE line of judgement (anything that needs action today).
Nothing else.
