clear all
set more off

cd ../..

********************************************************************

***Baseline
//Aggregate indexes
local sent_list lm gi lsd pc rpu
foreach sent of local sent_list {
	global index="`sent'"
	do "code/econometric_analysis/do_files/lp_aggregate"
}

//Categorical indexes
local sent_list lm gi lsd rpu
foreach sent of local sent_list {
	foreach i of num 1/14 {
		global index="`sent'_dda`i'"
		do "code/econometric_analysis/do_files/lp_area"
	}
}

***Quarterly analysis
local sent_list lm gi lsd rpu
foreach sent of local sent_list {
	global index="`sent'"
	do "code/econometric_analysis/do_files/lp_quarterly"
}
