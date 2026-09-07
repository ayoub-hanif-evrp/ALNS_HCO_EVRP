# Research questions (data-driven, post-HCO-fix)

Negative mean paired % = reduction. Results are reported as computed; no wording is softened.

## RQ1. Partial charging (ALNS-FR → ALNS-PR)
- charged_energy: original: mean %=-26.01 CI=[-33.42183345591779,-18.517305006858766] W/T/L=42/0/18; relaxed: mean %=-11.52 CI=[-15.751283819639704,-7.413315344064899] W/T/L=39/0/21
- charging_time: original: mean %=-25.43 CI=[-32.730707573442075,-17.9200037599723] W/T/L=40/0/20; relaxed: mean %=-10.74 CI=[-14.830032924961857,-6.804348689029286] W/T/L=40/0/20
- charging_expense: original: mean %=-26.82 CI=[-34.56785188684894,-19.0952286314324] W/T/L=44/0/16; relaxed: mean %=-12.62 CI=[-17.260635249898584,-8.137418608292172] W/T/L=40/0/20

## RQ2. Heterogeneous information on fixed routes (Detour → Heterogeneous)
- charging_expense: original: mean %=-0.37 CI=[-0.7127869231017523,-0.10499813722383246] W/T/L=7/53/0; relaxed: mean %=0.00 CI=[0.0,0.0] W/T/L=0/60/0
- station_waiting: original: mean %=-1.25 CI=[-2.2714923840757284,-0.3583556569228277] W/T/L=7/53/0; relaxed: mean %=0.00 CI=[0.0,0.0] W/T/L=0/60/0
- charging_time: original: mean %=0.78 CI=[0.19405124327121423,1.5087006940244387] W/T/L=0/53/7; relaxed: mean %=0.00 CI=[0.0,0.0] W/T/L=0/60/0
- charging_visits: original: mean %=0.00 CI=[0.0,0.0] W/T/L=0/60/0; relaxed: mean %=0.00 CI=[0.0,0.0] W/T/L=0/60/0

## RQ3. Is HCO useful beyond matched Phase-II search? (PhaseII-NoHCO → Heterogeneous)
- charging_expense: original: mean %=-3.66 CI=[-6.852782137726633,-0.9543296228769061] W/T/L=8/29/23; relaxed: mean %=0.32 CI=[0.14313737814649177,0.5271231380978744] W/T/L=0/39/21
- station_waiting: original: mean %=-3.04 CI=[-5.481547645124518,-0.9019323583815544] W/T/L=11/33/16; relaxed: mean %=0.21 CI=[0.09656756694838892,0.33284438197285476] W/T/L=1/44/15
- distance: original: mean %=-0.01 CI=[-0.03600277235425538,0.014517500081623594] W/T/L=8/24/28; relaxed: mean %=0.02 CI=[0.007272913911858058,0.02655011773769186] W/T/L=0/38/22

## RQ4. Does heterogeneity improve Phase II? (Detour → Heterogeneous)
- charging_expense: original: mean %=-0.53 CI=[-0.9508532548847235,-0.1601845487350296] W/T/L=8/52/0; relaxed: mean %=0.00 CI=[0.0,0.0] W/T/L=0/60/0
- station_waiting: original: mean %=-1.63 CI=[-2.8620096402329285,-0.5560862670705212] W/T/L=8/52/0; relaxed: mean %=0.00 CI=[0.0,0.0] W/T/L=0/60/0
- distance: original: mean %=0.00 CI=[-0.007691340805794001,0.01344968174814251] W/T/L=2/52/6; relaxed: mean %=0.00 CI=[0.0,0.0] W/T/L=0/60/0

## RQ5. What does Phase II add to the shared starting point? (X_PR_15 → PhaseII-Heterogeneous)
- charging_expense: original: mean %=-4.31 CI=[-7.601779306912976,-1.450424298386676] W/T/L=8/52/0; relaxed: mean %=0.00 CI=[0.0,0.0] W/T/L=0/60/0
- station_waiting: original: mean %=-3.28 CI=[-5.779427591804593,-1.092429768240253] W/T/L=8/52/0; relaxed: mean %=0.00 CI=[0.0,0.0] W/T/L=0/60/0
- distance: original: mean %=-0.03 CI=[-0.05669447411177439,-0.01241914608642609] W/T/L=8/52/0; relaxed: mean %=0.00 CI=[0.0,0.0] W/T/L=0/60/0

## RQ6. What stations does HCO actually choose?
See `outputs/tables/table_station_selection.csv` and `outputs/processed/charging_indicators.csv`.

## RQ7. When does HCO activate?
See `outputs/raw/hco_activity.csv` and `outputs/analysis/hco_activation_analysis.csv`.
**Caveat:** do **not** quote `improvement_rate`, `objective_ties`, or
insert/replace counts from the current CSV — see
`outputs/analysis/HCO_ACTIVITY_CAVEAT.md`. Solution-level results are unaffected.

## RQ6 family note
See `outputs/tables/table_family_results.csv`.

- Fixed-route original: mean P_used Detour=0.5165, Het=0.5116; W_used Detour=14.4205, Het=14.2645; r_used Detour=1.1942, Het=1.1861.
- Fixed-route relaxed: mean P_used Detour=0.5188, Het=0.5188; W_used Detour=14.4949, Het=14.4949; r_used Detour=1.1980, Het=1.1980.
