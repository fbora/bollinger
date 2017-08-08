\p 50117
\l u.q

tickers: `IBM`GOOG`FB`AMZN`AAPL
trade: ([]timestamp:`time$(); sym:`symbol$(); price:`float$());
.u.init[];

.z.ts:{
    row: enlist `timestamp`sym`price!(.z.T; first 1?tickers; 100+rand 10);
    `trade insert row
    .u.pub[`trade; row]
    0N! value flip row; };

\t 1000
