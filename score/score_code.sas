/*---------------------------------------------------------
  The options statement below should be placed
  before the data step when submitting this code.
---------------------------------------------------------*/
options VALIDMEMNAME=EXTEND VALIDVARNAME=ANY;
   /*------------------------------------------
   Generated SAS Scoring Code
     Date             : September 30, 2025, 11:34:11 PM
     Locale           : en_US
     Model Type       : Logistic Regression
     Interval variable: BAD
     Interval variable: CLAGE
     Interval variable: CLNO
     Interval variable: DEBTINC
     Interval variable: DELINQ
     Interval variable: DEROG
     Interval variable: LOAN
     Interval variable: MORTDUE
     Interval variable: NINQ
     Interval variable: VALUE
     Interval variable: YOJ
     Class variable   : REASON
     Class variable   : va__d__E_JOB(JOB)
     Response variable: va__d__E_JOB(JOB)
     Distribution     : Binary
     Link Function    : Logit
     ------------------------------------------*/
/* Temporary Computed Columns */
if (('JOB'n = 'Self')) then do;
'va__d__E_JOB'n= 1.0;
end;
else do;
'va__d__E_JOB'n= 0.0;
end;
;

/*------------------------------------------*/
   /*---------------------------------------------------------
     Generated SAS Scoring Code
     Date: 30Sep2025:23:34:10
     -------------------------------------------------------*/

   /*---------------------------------------------------------
   Defining temporary arrays and variables   
     -------------------------------------------------------*/
   drop _badval_ _linp_ _temp_ _i_ _j_;
   _badval_ = 0;
   _linp_   = 0;
   _temp_   = 0;
   _i_      = 0;
   _j_      = 0;
   drop MACLOGBIG;
   MACLOGBIG= 7.0978271289338392e+02;

   array _xrow_620_0_{14} _temporary_;
   array _beta_620_0_{14} _temporary_ (   -6.63907523027995
          -1.34395880433063
                          0
           0.58922327656478
            0.0029475261967
           0.06138379668666
           0.01275269497189
          -0.18760742440544
          -0.16422123036655
           0.00005898451537
          -0.00001492069973
            0.0023184323174
           0.00001645099895
          -0.12352970853956);

   length _REASON_ $7; drop _REASON_;
   _REASON_ = left(trim(put(REASON,$7.)));
   /*------------------------------------------*/
   /*Missing values in model variables result  */
   /*in missing values for the prediction.     */
   if missing(CLNO) 
      or missing(NINQ) 
      or missing(MORTDUE) 
      or missing(DEROG) 
      or missing(VALUE) 
      or missing(BAD) 
      or missing(DELINQ) 
      or missing(LOAN) 
      or missing(YOJ) 
      or missing(DEBTINC) 
      or missing(CLAGE) 
      then do;
         _badval_ = 1;
         goto skip_620_0;
   end;
   /*------------------------------------------*/

   do _i_=1 to 14; _xrow_620_0_{_i_} = 0; end;

   _xrow_620_0_[1] = 1;

   _temp_ = 1;
   select (_REASON_);
      when ('DebtCon') _xrow_620_0_[2] = _temp_;
      when ('HomeImp') _xrow_620_0_[3] = _temp_;
      otherwise do; _badval_ = 1; goto skip_620_0; end;
   end;

   _xrow_620_0_[4] = BAD;

   _xrow_620_0_[5] = CLAGE;

   _xrow_620_0_[6] = CLNO;

   _xrow_620_0_[7] = DEBTINC;

   _xrow_620_0_[8] = DELINQ;

   _xrow_620_0_[9] = DEROG;

   _xrow_620_0_[10] = LOAN;

   _xrow_620_0_[11] = MORTDUE;

   _xrow_620_0_[12] = NINQ;

   _xrow_620_0_[13] = VALUE;

   _xrow_620_0_[14] = YOJ;

   do _i_=1 to 14;
      _linp_ + _xrow_620_0_{_i_} * _beta_620_0_{_i_};
   end;

   skip_620_0:
   length I_va__d__E_JOB $12;
   label I_va__d__E_JOB = 'Into: va__d__E_JOB';
   array _levels_620_{2} $ 12 _TEMPORARY_ ('1'
   ,'0'
   );
   label P_va__d__E_JOB1 = 'Predicted: va__d__E_JOB=1';
   if (_badval_ eq 0) and not missing(_linp_) then do;
      if (_linp_ > 0) then do;
         P_va__d__E_JOB1 = 1 / (1+exp(-_linp_));
      end; else do;
         P_va__d__E_JOB1 = exp(_linp_) / (1+exp(_linp_));
      end;
      P_va__d__E_JOB0 = 1 - P_va__d__E_JOB1;
      if P_va__d__E_JOB1 >= 0.5                  then do;
         I_va__d__E_JOB = _levels_620_{1};
      end; else do;
         I_va__d__E_JOB = _levels_620_{2};
      end;
   end; else do;
      _linp_ = .;
      P_va__d__E_JOB1 = .;
      P_va__d__E_JOB0 = .;
      I_va__d__E_JOB = ' ';
   end;
   /*------------------------------------------*/
   /*_VA_DROP*/ drop 'va__d__E_JOB'n 'I_va__d__E_JOB'n 'P_va__d__E_JOB1'n 'P_va__d__E_JOB0'n;
length 'I_va__d__E_JOB_620'n $32;
      'I_va__d__E_JOB_620'n='I_va__d__E_JOB'n;
'P_va__d__E_JOB1_620'n='P_va__d__E_JOB1'n;
'P_va__d__E_JOB0_620'n='P_va__d__E_JOB0'n;
   /*------------------------------------------*/
