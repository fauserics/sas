%put WORK=%sysfunc(pathname(work));
data _null_; file "%sysfunc(pathname(work))/stmp_test.txt"; put "ok"; run;
