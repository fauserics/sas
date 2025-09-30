%let RAW_URL=https://raw.githubusercontent.com/fauserics/sas/main/data/hmeq.csv;

proc http
  url="&RAW_URL."
  method="GET"
  out="hmeq.csv";
run;


