clear all
set more off

cd ..

********************************************************************
***Baseline
//Aggregate Sentiment
global index="lm"
do "lp_sent"

//Aggregate RPU
do "lp_rpu"

//Categorical Sentiment
foreach i of num 1/14 {
	global index="lm_dda`i'"
	do "lp_sent_area"
}

//Categorical RPU
foreach i of num 1/14 {
	global index="rpu_dda`i'"
	do "lp_rpu_area"
}

***Alternative sentiment indexes
//Aggregate Sentiment
local sent_list gi lsd pc
foreach sent of local sent_list {
	global index="`sent'"
	do "lp_sent"
}

//Categorical Sentiment
local sent_list gi_dda lsd_dda
foreach area of local sent_list {
	foreach i of num 1/14 {
		global index="`area'`i'"
		do "lp_sent_area"
	}
}
