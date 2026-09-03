PYTHON := .venv/bin/python

.PHONY: help daily close hot forward trades trade-open trade-summary py

help:
	@echo "常用命令："
	@echo "  make daily                 正式：刷盘中价并生成交易 HTML"
	@echo "  make close                 收盘：更新 K 线 + 观察增量入簿（不生成 HTML）"
	@echo "  make hot                   只生成热门板块"
	@echo "  make forward ARGS='...'    选股后续收益报告"
	@echo "  make trades ARGS='...'     交易台账命令"
	@echo "  make trade-open            查看当前持仓"
	@echo "  make trade-summary         查看交易汇总"
	@echo "  make py SCRIPT=x.py ARGS='...'  运行任意 Python 脚本"

daily:
	$(PYTHON) run_daily_pool.py $(ARGS)

close:
	$(PYTHON) run_daily_pool.py --close $(ARGS)

hot:
	$(PYTHON) run_daily_pool.py --hot-only $(ARGS)

forward:
	$(PYTHON) pool_forward_returns.py $(ARGS)

trades:
	$(PYTHON) trades_log.py $(ARGS)

trade-open:
	$(PYTHON) trades_log.py open

trade-summary:
	$(PYTHON) trades_log.py summary

py:
	@test -n "$(SCRIPT)" || (echo "缺少 SCRIPT，例如：make py SCRIPT=short_burst_backtest.py"; exit 2)
	$(PYTHON) $(SCRIPT) $(ARGS)
