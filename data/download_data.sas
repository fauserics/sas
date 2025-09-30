%let RAW_URL=https://raw.githubusercontent.com/fauserics/sas/main/data/hmeq.csv;

filename hmeq "%sysfunc(pathname(work))/hmeq.csv";
proc http
  url="&RAW_URL."
  method="GET"
  out=hmeq;
run;

/* Verificar que se descargó */
%put NOTE: Existe hmeq.csv?  %sysfunc(fexist(hmeq));
