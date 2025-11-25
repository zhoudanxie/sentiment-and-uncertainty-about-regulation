clear
graph drop _all
cap drop all

*Basic data set-up
use data/processed_data/data_for_analysis/regindex,clear

*Michigan consumer sentiment data - some robustness test data
merge 1:1 year month using data/processed_data/data_for_analysis/consumer_sentiment_data,keep(1 3) nogen
ren Month mich

***Merge in Stock data
merge m:1 year month using data/processed_data/data_for_analysis/sp_500_data, keep(1 3) nogen

***Merge in Macro data 
merge m:1 year month using data/processed_data/data_for_analysis/macro_data, keep(1 3) nogen
replace vix=. if year<1990

***Merge in BBD EPU 
merge 1:1 year month using data/processed_data/data_for_analysis/epu,keep(1 3) nogen

***Merge in Shapiro sentiment
merge 1:1 year month using data/processed_data/data_for_analysis/newssent,keep(1 3) nogen

***Transform data
gen lgdp=log(gdp)

gen ym=year+(month-1)/12
replace quarter=floor((month+2)/3) if quarter==.
gen yq=year+(quarter-1)/4
gen time=ym*12
tsset time
gen lemp=log(employment)
gen lip=log(indus)
gen lsp=log(sp)
gen lvix=log(vix)

ren sp sp
ren fedfundsrate ffr

ren RegRelevance rri
ren UncertaintyIndex rpu
ren GIIndex gi
ren LSDIndex lsd
ren LMIndex lm
ren SentimentPC1_standardized pc
sum rpu lm gi lsd

*For monthly, output is industrial production
replace lgdp=lip

* Generate interaction term
gen sentXrpu=lm*rpu

* Define high and low sentiment state
sum lm
global highstate=r(mean)+r(sd)
global lowstate=r(mean)-r(sd)

* Choose impulse response horizon
local hmax = 12

* Define a shock
sum rpu
global ratio=r(sd)	//a positive std shock

* Select lags
varsoc rpu lsp ffr lemp lgdp, maxlag(10)

/* Generate LHS variables for the LPs */

* levels
forvalues h = 0/`hmax' {
	gen lgdp_`h' = f`h'.lgdp 
}

forvalues h = 0/`hmax' {
	gen lemp_`h' = f`h'.lemp 
}

/* Run the LPs */
* Levels - lgdp
eststo clear
cap drop b_high u90_high d90_high u95_high d95_high b_low u90_low d90_low u95_low d95_low Years Zero
gen Years = _n-1 if _n<=`hmax'+1
gen Zero =  0    if _n<=`hmax'+1
gen b_high=0
gen u90_high=0
gen d90_high=0
gen u95_high=0
gen d95_high=0
gen b_low=0
gen u90_low=0
gen d90_low=0
gen u95_low=0
gen d95_low=0
forv h = 0/`hmax' {
	* levels
	reg lgdp_`h' l(0/3).sentXrpu l(0/3).lm l(0/3).rpu l(0/3).lsp l(0/3).ffr l(0/3).lemp l(0/3).lgdp, vce(robust)

	lincom rpu+sentXrpu*$highstate, level(90)
	replace b_high = r(estimate)*$ratio*100 if _n == `h'+1
	replace u90_high = (r(estimate)+1.645*r(se))*$ratio*100  if _n == `h'+1
	replace d90_high = (r(estimate)-1.645*r(se))*$ratio*100  if _n == `h'+1
	lincom rpu+sentXrpu*$highstate, level(95)
	replace u95_high = (r(estimate)+1.96*r(se))*$ratio*100  if _n == `h'+1
	replace d95_high = (r(estimate)-1.96*r(se))*$ratio*100  if _n == `h'+1

	lincom rpu+sentXrpu*$lowstate, level(90)
	replace b_low = r(estimate)*$ratio*100 if _n == `h'+1
	replace u90_low = (r(estimate)+1.645*r(se))*$ratio*100  if _n == `h'+1
	replace d90_low = (r(estimate)-1.645*r(se))*$ratio*100  if _n == `h'+1
	lincom rpu+sentXrpu*$lowstate, level(95)
	replace u95_low = (r(estimate)+1.96*r(se))*$ratio*100  if _n == `h'+1
	replace d95_low = (r(estimate)-1.96*r(se))*$ratio*100  if _n == `h'+1

	eststo
}


preserve
keep Years b_high u90_high d90_high u95_high d95_high b_low u90_low d90_low u95_low d95_low
keep if Years!=.
save "output/rpu_lgdp_interaction.dta",replace
restore

* Levels - lemp
eststo clear
cap drop b_high u90_high d90_high u95_high d95_high b_low u90_low d90_low u95_low d95_low Years Zero
gen Years = _n-1 if _n<=`hmax'+1
gen Zero =  0    if _n<=`hmax'+1
gen b_high=0
gen u90_high=0
gen d90_high=0
gen u95_high=0
gen d95_high=0
gen b_low=0
gen u90_low=0
gen d90_low=0
gen u95_low=0
gen d95_low=0
forv h = 0/`hmax' {
	* levels
	reg lemp_`h' l(0/3).sentXrpu l(0/3).lm l(0/3).rpu l(0/3).lsp l(0/3).ffr l(0/3).lemp l(0/3).lgdp, vce(robust)

	lincom rpu+sentXrpu*$highstate, level(90)
	replace b_high = r(estimate)*$ratio*100 if _n == `h'+1
	replace u90_high = (r(estimate)+1.645*r(se))*$ratio*100  if _n == `h'+1
	replace d90_high = (r(estimate)-1.645*r(se))*$ratio*100  if _n == `h'+1
	lincom rpu+sentXrpu*$highstate, level(95)
	replace u95_high = (r(estimate)+1.96*r(se))*$ratio*100  if _n == `h'+1
	replace d95_high = (r(estimate)-1.96*r(se))*$ratio*100  if _n == `h'+1

	lincom rpu+sentXrpu*$lowstate, level(90)
	replace b_low = r(estimate)*$ratio*100 if _n == `h'+1
	replace u90_low = (r(estimate)+1.645*r(se))*$ratio*100  if _n == `h'+1
	replace d90_low = (r(estimate)-1.645*r(se))*$ratio*100  if _n == `h'+1
	lincom rpu+sentXrpu*$lowstate, level(95)
	replace u95_low = (r(estimate)+1.96*r(se))*$ratio*100  if _n == `h'+1
	replace d95_low = (r(estimate)-1.96*r(se))*$ratio*100  if _n == `h'+1

	eststo
}

preserve
keep Years b_high u90_high d90_high u95_high d95_high b_low u90_low d90_low u95_low d95_low
keep if Years!=.
save "output/rpu_lemp_interaction.dta",replace
restore

 
/* THE END */
