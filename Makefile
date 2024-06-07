

.PHY: download
download:
	@pairs=$(or ${pairs}, BTC/USDT); \
	days=$(or ${days}, 30); \
	timeframe=$(or ${timeframe}, 1h); \
	docker-compose run --rm freqtrade download-data --pairs $$pairs --exchange binance --days $$days -t $$timeframe


.PHY: backtest
backtest:
	docker-compose run --rm freqtrade backtesting --strategy-path user_data/strategies/ --strategy-list "TestStrategy" -i 1h --timerange 20240310-


# Prevent make from trying to interpret the command-line arguments as Make targets
%:
	@:
