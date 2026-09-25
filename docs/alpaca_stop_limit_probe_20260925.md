
## Probe run 2026-09-25T17:08:46.639640+00:00

```
[17:07:42.055 UTC] latest AAPL quote bid 339.6 ask 339.65 (probe prices 15 % above the ask cannot fill)
[17:07:42.055 UTC] PLACE AAPL qty 1 buy stop-limit stop=limit=$390.60 (ask $339.65) client_order_id onemil-probe-1790356062055-q1
[17:07:42.428 UTC]   accepted in 0.37s: {"id": "7fac7eba-f696-452e-8b70-08e2bb8ac00e", "client_order_id": "onemil-probe-1790356062055-q1", "status": "OrderStatus.PENDING_NEW", "type": "OrderType.STOP_LIMIT", "order_type": "OrderType.STOP_LIMIT", "side": "OrderSide.BUY", "qty": "1", "stop_price": "390.6", "limit_price": "390.6", "time_in_force": "TimeInForce.DAY", "submitted_at": "2026-09-25 17:07:42.357799+00:00", "updated_at": "2026-09-25 17:07:42.358240+00:00", "filled_qty": "0"}
[17:07:42.537 UTC]   status OrderStatus.NEW after 0.11s
[17:07:52.884 UTC] REPLACE stop/limit -> $390.70
[17:07:52.993 UTC]   replace returned in 0.11s: {"id": "38509c80-07af-49cc-a391-8c873a7f599c", "client_order_id": "10ed4088-173b-48b1-b37d-db48b45ba491", "status": "OrderStatus.NEW", "type": "OrderType.STOP_LIMIT", "order_type": "OrderType.STOP_LIMIT", "side": "OrderSide.BUY", "qty": "1", "stop_price": "390.7", "limit_price": "390.7", "time_in_force": "TimeInForce.DAY", "submitted_at": "2026-09-25 17:07:52.938675+00:00", "updated_at": "2026-09-25 17:07:52.939146+00:00", "replaces": "7fac7eba-f696-452e-8b70-08e2bb8ac00e", "filled_qty": "0"} (id changed: True)
[17:07:52.993 UTC] CANCEL
[17:07:53.210 UTC]   status OrderStatus.CANCELED after 0.11s
[17:08:13.409 UTC]   cancel confirmed in 20.42s: {"id": "38509c80-07af-49cc-a391-8c873a7f599c", "client_order_id": "10ed4088-173b-48b1-b37d-db48b45ba491", "status": "OrderStatus.CANCELED", "type": "OrderType.STOP_LIMIT", "order_type": "OrderType.STOP_LIMIT", "side": "OrderSide.BUY", "qty": "1", "stop_price": "390.7", "limit_price": "390.7", "time_in_force": "TimeInForce.DAY", "submitted_at": "2026-09-25 17:07:52.940228+00:00", "updated_at": "2026-09-25 17:07:53.069509+00:00", "canceled_at": "2026-09-25 17:07:53.053000+00:00", "replaces": "7fac7eba-f696-452e-8b70-08e2bb8ac00e", "filled_qty": "0"}
[17:08:13.409 UTC]   WARNING: order not confirmed canceled — check the dashboard for prefix onemil-probe-
[17:08:14.413 UTC] PLACE AAPL qty 7 buy stop-limit stop=limit=$390.60 (ask $339.65) client_order_id onemil-probe-1790356094413-q7
[17:08:14.534 UTC]   accepted in 0.12s: {"id": "5bfa78d2-de07-4244-8487-fb596f1f5650", "client_order_id": "onemil-probe-1790356094413-q7", "status": "OrderStatus.PENDING_NEW", "type": "OrderType.STOP_LIMIT", "order_type": "OrderType.STOP_LIMIT", "side": "OrderSide.BUY", "qty": "7", "stop_price": "390.6", "limit_price": "390.6", "time_in_force": "TimeInForce.DAY", "submitted_at": "2026-09-25 17:08:14.470499+00:00", "updated_at": "2026-09-25 17:08:14.470864+00:00", "filled_qty": "0"}
[17:08:14.643 UTC]   status OrderStatus.NEW after 0.11s
[17:08:25.112 UTC] REPLACE stop/limit -> $390.70
[17:08:25.221 UTC]   replace returned in 0.11s: {"id": "153a2b17-c268-43bc-b9d4-114bc2910df3", "client_order_id": "224a4c56-37c9-48dd-be21-e6cabb73c369", "status": "OrderStatus.NEW", "type": "OrderType.STOP_LIMIT", "order_type": "OrderType.STOP_LIMIT", "side": "OrderSide.BUY", "qty": "7", "stop_price": "390.7", "limit_price": "390.7", "time_in_force": "TimeInForce.DAY", "submitted_at": "2026-09-25 17:08:25.166481+00:00", "updated_at": "2026-09-25 17:08:25.166968+00:00", "replaces": "5bfa78d2-de07-4244-8487-fb596f1f5650", "filled_qty": "0"} (id changed: True)
[17:08:25.221 UTC] CANCEL
[17:08:25.436 UTC]   status OrderStatus.CANCELED after 0.11s
[17:08:45.529 UTC]   cancel confirmed in 20.31s: {"id": "153a2b17-c268-43bc-b9d4-114bc2910df3", "client_order_id": "224a4c56-37c9-48dd-be21-e6cabb73c369", "status": "OrderStatus.CANCELED", "type": "OrderType.STOP_LIMIT", "order_type": "OrderType.STOP_LIMIT", "side": "OrderSide.BUY", "qty": "7", "stop_price": "390.7", "limit_price": "390.7", "time_in_force": "TimeInForce.DAY", "submitted_at": "2026-09-25 17:08:25.168255+00:00", "updated_at": "2026-09-25 17:08:25.295225+00:00", "canceled_at": "2026-09-25 17:08:25.279000+00:00", "replaces": "5bfa78d2-de07-4244-8487-fb596f1f5650", "filled_qty": "0"}
[17:08:45.529 UTC]   WARNING: order not confirmed canceled — check the dashboard for prefix onemil-probe-
[17:08:46.639 UTC] open orders with prefix onemil-probe-: 0
[17:08:46.639 UTC] DONE
```
