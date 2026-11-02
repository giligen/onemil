---
name: books
description: Status of every strategy book (service, HOD/ORB paper books, MOM sleeve, rails, crons) — run when the owner asks how the strategies are doing. Named `books` because `/status` is a built-in Claude Code command.
---

Run `python3 scripts/book_status.py` from /home/ec2-user/onemil (read-only; takes ~15-20 s because it greps today's
journal). Relay the output verbatim to the owner, then add ONE line of judgement (anything that needs action today).
Nothing else.

Then build the HTML report: `python3 scripts/book_status_html.py --out reports/book_status_$(date +%Y%m%d).html`
(it runs `book_status.py --json` itself; prints the path and size). Publish that file with the Artifact tool: stable
title `OneMil Books`, icon `chart`, and redeploy the SAME artifact URL every day (the URL lives in
`reports/BOOKS_ARTIFACT_URL`; create the file on the first publish). Give the owner the link after the text report.
