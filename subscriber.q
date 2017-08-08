\p 5018
\l u.q

upd:{[tblname; tbldata]
    0N! value flip tbldata;
    tblname insert tbldata;};
    
trade: ([] timestamp: `time$(); sym:`symbol$(); price: `float$())

h: hopen`:localhost:5017

h(`.u.sub;`trade;`AAPL`AMZN);
