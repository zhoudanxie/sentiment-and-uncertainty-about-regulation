clear all
set more off

cd ../..

********************************************************************
***Baseline
//Aggregate Sentiment
global index="lm"
do "code/econometric_analysis/lp_sent"

//Aggregate RPU
do "code/econometric_analysis/lp_rpu"

//Categorical Sentiment
foreach i of num 1/14 {
	global index="lm_dda`i'"
	do "code/econometric_analysis/lp_sent_area"
}

//Categorical RPU
foreach i of num 1/14 {
	global index="rpu_dda`i'"
	do "code/econometric_analysis/lp_rpu_area"
}

***Alternative sentiment indexes
//Aggregate Sentiment
local sent_list gi lsd pc
foreach sent of local sent_list {
	global index="`sent'"
	do "code/econometric_analysis/lp_sent"
}

//Categorical Sentiment
local sent_list gi_dda lsd_dda
foreach area of local sent_list {
	foreach i of num 1/14 {
		global index="`area'`i'"
		do "code/econometric_analysis/lp_sent_area"
	}
}
