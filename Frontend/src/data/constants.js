export const LIVE_DATA = {
  hour: 4320,
  timestamp: "2023-06-15T14:00:00",
  memory_norm: 0.7231,
  wind_metrics: {
    avg_ws_24h: 7.412,
    avg_power_proxy_24h: 0.384,
    high_wind_hours_24h: 8,
    low_wind_hours_24h: 3,
    mean_pred_error: 0.002841,
  },
  recent_errors: [
    0.0031, 0.0028, 0.0024, 0.0029, 0.0033, 0.0027, 0.0031, 0.0025,
    0.0028, 0.003, 0.0026, 0.0029, 0.0031, 0.0027, 0.0024, 0.0028,
    0.003, 0.0026, 0.0029, 0.0031, 0.0025, 0.0028, 0.0027, 0.003,
  ],
  features: {
    WS50M: { predicted: 8.1241, actual: 7.9832, error: 0.1409 },
    WS10M: { predicted: 6.3412, actual: 6.2103, error: 0.1309 },
    wind_power_50m: { predicted: 0.4123, actual: 0.3941, error: 0.0182 },
    wind_power_10m: { predicted: 0.2741, actual: 0.2512, error: 0.0229 },
    T2M: { predicted: 12.3241, actual: 12.1032, error: 0.2209 },
    ALLSKY_SFC_SW_DWN: { predicted: 145.23, actual: 142.81, error: 2.42 },
    RH2M: { predicted: 78.41, actual: 79.1, error: -0.69 },
    PS: { predicted: 1013.23, actual: 1013.51, error: -0.28 },
    wind_shear_ratio: { predicted: 0.8231, actual: 0.8412, error: -0.0181 },
    memory_norm_last_hour: { predicted: 0.7181, actual: 0.7312, error: -0.0131 },
    WD50M: { predicted: 234.12, actual: 231.89, error: 2.23 },
    CLOUD_AMT: { predicted: 62.31, actual: 64.12, error: -1.81 },
    RHOA: { predicted: 1.2341, actual: 1.2289, error: 0.0052 },
  },
};

export const MONTHLY_DATA = [
  {
    month: "2023-01",
    key_numbers: {
      wind_speed_avg_ms: 9.241,
      capacity_factor_pct: 28.41,
      est_energy_mwh: 21241.8,
      est_revenue_cfd_gbp: 2081696,
      est_gross_profit_gbp: 1581696,
      lost_revenue_low_wind_gbp: 91980,
      carbon_avoided_tco2e: 4949.3,
      overall_risk_rating: "MEDIUM",
      high_wind_hours_gt12ms: 187,
      low_wind_hours_lt4ms: 31,
      calm_fraction_pct: 4.2,
      bdh_memory_norm_avg: 0.7814,
    },
    llm_conclusion: {
      headline: "January delivered above-average wind conditions with several high-output storm events driving strong generation.",
      overall_risk_rating: "MEDIUM",
      sections: {
        "1_weather_and_wind_summary": "January 2023 was a notably windy month for the UK, with average wind speeds of 9.24 m/s at 50m hub height — approximately 18% above the long-term seasonal mean of 7.8 m/s. The BDH model recorded 187 high-wind hours exceeding 12 m/s, including two significant storm events mid-month. Low-wind periods were limited to just 31 hours (4.2% calm fraction), indicating strong and consistent resource availability. The BDH memory norm averaged 0.78, suggesting a stable and predictable wind regime.",
        "2_energy_generation": "Using the BDH-derived average wind speed of 9.24 m/s, the estimated capacity factor for January is 28.41%. Calculation: CF = min(0.45, 0.45 × (9.24 / 12.0)³) × 0.97 availability = 0.2841. Estimated monthly energy = 0.2841 × 210 MW × 744 hours = 21,241.8 MWh. This is below SSE's onshore target range of 35–40% annual CF, consistent with January's sub-rated wind speeds.",
        "3_financial_performance": "Estimated CfD revenue = 21,241.8 MWh × £98/MWh = £2,081,696. Monthly O&M cost = £500,000. Estimated gross profit = £2,081,696 − £500,000 = £1,581,696. Lost revenue from low-wind hours = £91,980. Carbon avoided = 4,949.3 tCO2e.",
        "4_climate_risk_tcfd": "Physical climate risk for January is rated MEDIUM. Positive factors: wind resource was 18% above seasonal norm. Risk factors: two storm events exceeded 18 m/s, approaching cut-out thresholds. Per SSE's TCFD 2023 disclosure, increased storm frequency under 2°C warming is classified as a high-likelihood, medium-impact physical risk.",
        "5_strategic_alignment": "January's generation contributes positively to SSE's net zero 2050 pathway. The CfD revenue model aligns with SSE's strategy of securing long-term contracted revenues to underpin the £18bn low-carbon investment programme.",
        "6_recommended_actions": "1. STORM PREPAREDNESS: Review turbine storm control settings ahead of Feb–March storm season.\n2. FORECASTING OPTIMISATION: Leverage high memory norm (0.78) to improve day-ahead scheduling.\n3. LOW-WIND HEDGING: Explore battery storage options for calm periods.",
      },
    },
  },
  {
    month: "2023-02",
    key_numbers: {
      wind_speed_avg_ms: 7.831,
      capacity_factor_pct: 17.23,
      est_energy_mwh: 11547.2,
      est_revenue_cfd_gbp: 1131625,
      est_gross_profit_gbp: 631625,
      lost_revenue_low_wind_gbp: 181230,
      carbon_avoided_tco2e: 2690.5,
      overall_risk_rating: "HIGH",
      high_wind_hours_gt12ms: 98,
      low_wind_hours_lt4ms: 61,
      calm_fraction_pct: 9.1,
      bdh_memory_norm_avg: 0.6421,
    },
    llm_conclusion: {
      headline: "February saw significantly weaker wind with extended calm periods, resulting in below-target generation and elevated low-wind revenue risk.",
      overall_risk_rating: "HIGH",
      sections: {
        "1_weather_and_wind_summary": "February 2023 delivered below-average wind conditions, with a monthly mean of 7.83 m/s — approximately 8% below seasonal norm. A 4-day anticyclonic blocking event in the second week drove sustained calm conditions across northern England and Scotland. The BDH memory norm of 0.64 reflects a more volatile and less predictable regime.",
        "2_energy_generation": "Estimated capacity factor: CF = 0.1723. Estimated monthly energy = 11,547.2 MWh — a 45.6% decline from January. The blocking event alone accounts for approximately 3,200 MWh of lost generation.",
        "3_financial_performance": "Estimated CfD revenue = £1,131,625. Monthly O&M = £500,000. Gross profit = £631,625. Lost revenue from low-wind = £181,230. February gross profit is 60% below January's.",
        "4_climate_risk_tcfd": "Physical climate risk is rated HIGH. The anticyclonic blocking event represents a key physical risk scenario identified in SSE's TCFD 2023 report under 'wind drought' conditions.",
        "5_strategic_alignment": "February highlights the importance of SSE's portfolio diversification strategy across onshore, offshore, hydro and solar to hedge against wind drought events.",
        "6_recommended_actions": "1. WIND DROUGHT CONTINGENCY: Develop playbook for anticyclonic blocking events.\n2. PORTFOLIO HEDGING: Accelerate Dogger Bank offshore integration.\n3. BDH FORECASTING REVIEW: Recalibrate models ahead of spring.",
      },
    },
  },
  {
    month: "2023-03",
    key_numbers: {
      wind_speed_avg_ms: 8.614,
      capacity_factor_pct: 23.81,
      est_energy_mwh: 17747.3,
      est_revenue_cfd_gbp: 1739235,
      est_gross_profit_gbp: 1239235,
      lost_revenue_low_wind_gbp: 139482,
      carbon_avoided_tco2e: 4135.1,
      overall_risk_rating: "MEDIUM",
      high_wind_hours_gt12ms: 134,
      low_wind_hours_lt4ms: 47,
      calm_fraction_pct: 6.3,
      bdh_memory_norm_avg: 0.7102,
    },
    llm_conclusion: {
      headline: "March showed a recovery in wind resource from February's low, with improving capacity factors and moderate climate risk.",
      overall_risk_rating: "MEDIUM",
      sections: {
        "1_weather_and_wind_summary": "March 2023 marked a recovery with average wind speeds rising to 8.61 m/s. Two frontal systems drove high-wind periods while a brief anticyclone mid-month caused a short calm spell. BDH memory norm of 0.71 reflects moderate predictability.",
        "2_energy_generation": "Estimated capacity factor: CF = 0.2381. Estimated monthly energy = 17,747.3 MWh — a 53.7% increase over February.",
        "3_financial_performance": "Estimated CfD revenue = £1,739,235. Gross profit = £1,239,235. March gross profit is 96% of January's despite lower average wind, due to a full 744-hour month.",
        "4_climate_risk_tcfd": "Physical climate risk is rated MEDIUM. Wind variability is elevated (std dev 3.91 m/s) consistent with spring transition season instability.",
        "5_strategic_alignment": "Q1 2023 total carbon avoided: 11,775 tCO2e — on track for SSE's annual targets.",
        "6_recommended_actions": "1. SPRING MAINTENANCE WINDOW: Optimal conditions for scheduled turbine maintenance.\n2. Q1 PERFORMANCE REVIEW: Average CF of 23.2% is below 35% target.\n3. MEMORY NORM MONITORING: Trend 0.78→0.64→0.71 warrants attention.",
      },
    },
  },
  {
    month: "2023-04",
    key_numbers: {
      wind_speed_avg_ms: 7.124,
      capacity_factor_pct: 11.24,
      est_energy_mwh: 8032.1,
      est_revenue_cfd_gbp: 787145,
      est_gross_profit_gbp: 287145,
      lost_revenue_low_wind_gbp: 252210,
      carbon_avoided_tco2e: 1871.5,
      overall_risk_rating: "HIGH",
      high_wind_hours_gt12ms: 62,
      low_wind_hours_lt4ms: 85,
      calm_fraction_pct: 11.8,
      bdh_memory_norm_avg: 0.5931,
    },
    llm_conclusion: {
      headline: "April was the weakest month of Q2 — persistent anticyclones drove the lowest gross profit of 2023 at near-breakeven.",
      overall_risk_rating: "HIGH",
      sections: {
        "1_weather_and_wind_summary": "April 2023 dominated by a persistent high-pressure system for 12 of 30 days. Calm fraction reached 11.8% with 85 low-wind hours. BDH memory norm of 0.59 is the lowest recorded — high forecast uncertainty.",
        "2_energy_generation": "Estimated capacity factor: CF = 0.1124. Estimated monthly energy = 8,032.1 MWh — only 37.8% of January's output.",
        "3_financial_performance": "CfD revenue = £787,145. O&M = £500,000. Gross profit = £287,145. O&M consumes 63.5% of revenue — fleet approaching breakeven.",
        "4_climate_risk_tcfd": "Physical climate risk is rated HIGH. The 12-day blocking event is consistent with the 'wind drought' physical risk scenario in SSE's TCFD 2023 report.",
        "5_strategic_alignment": "April demonstrates the most acute version of wind concentration risk that SSE's diversification strategy is designed to address.",
        "6_recommended_actions": "1. WIND DROUGHT RESPONSE PLAN: Formalise protocol for multi-day blocking events.\n2. O&M COST REVIEW: Fixed costs consuming 63.5% of revenue — review deferrable items.\n3. SCENARIO PLANNING: Quantify 15–25% increase in summer blocking frequency under 2°C.",
      },
    },
  },
  {
    month: "2023-05",
    key_numbers: {
      wind_speed_avg_ms: 8.021,
      capacity_factor_pct: 19.81,
      est_energy_mwh: 14779.5,
      est_revenue_cfd_gbp: 1448391,
      est_gross_profit_gbp: 948391,
      lost_revenue_low_wind_gbp: 163290,
      carbon_avoided_tco2e: 3443.6,
      overall_risk_rating: "MEDIUM",
      high_wind_hours_gt12ms: 108,
      low_wind_hours_lt4ms: 55,
      calm_fraction_pct: 7.4,
      bdh_memory_norm_avg: 0.6812,
    },
    llm_conclusion: {
      headline: "May showed a moderate recovery from April's wind drought with improving generation and return to medium risk.",
      overall_risk_rating: "MEDIUM",
      sections: {
        "1_weather_and_wind_summary": "May 2023 partial recovery with average speeds rising to 8.02 m/s and calm fraction falling to 7.4%. Atlantic frontal systems in the second half drove the recovery.",
        "2_energy_generation": "Capacity factor: 0.1981. Estimated energy = 14,779.5 MWh — 84% improvement over April.",
        "3_financial_performance": "CfD revenue = £1,448,391. Gross profit = £948,391. O&M now represents 34.5% of revenue vs April's 63.5%.",
        "4_climate_risk_tcfd": "Physical climate risk rated MEDIUM. No storm events exceeded operational thresholds.",
        "5_strategic_alignment": "May supports Q2 generation targets. Gross profit margin recovery demonstrates importance of wind resource management.",
        "6_recommended_actions": "1. SUMMER PREPARATION: Review curtailment protocols for grid congestion.\n2. FORECASTING CALIBRATION: Improving memory norm — recalibrate scheduling models.\n3. H1 PERFORMANCE REVIEW: Average CF of 20.1% lags the 35–40% annual target.",
      },
    },
  },
  {
    month: "2023-06",
    key_numbers: {
      wind_speed_avg_ms: 6.512,
      capacity_factor_pct: 7.62,
      est_energy_mwh: 5488.1,
      est_revenue_cfd_gbp: 537833,
      est_gross_profit_gbp: 37833,
      lost_revenue_low_wind_gbp: 302820,
      carbon_avoided_tco2e: 1278.7,
      overall_risk_rating: "HIGH",
      high_wind_hours_gt12ms: 18,
      low_wind_hours_lt4ms: 102,
      calm_fraction_pct: 14.2,
      bdh_memory_norm_avg: 0.5124,
    },
    llm_conclusion: {
      headline: "June was the weakest month of 2023 — summer anticyclonic dominance drove near-breakeven economics and peak low-wind seasonal risk.",
      overall_risk_rating: "HIGH",
      sections: {
        "1_weather_and_wind_summary": "June 2023 was the most challenging month. Average wind speeds fell to 6.51 m/s with calm fraction of 14.2% and 102 low-wind hours. Classic summer anticyclonic dominance suppressed Atlantic frontal activity. BDH memory norm of 0.51 is the lowest recorded.",
        "2_energy_generation": "Capacity factor: 0.0762. Estimated energy = 5,488.1 MWh — lowest monthly figure in the dataset.",
        "3_financial_performance": "CfD revenue = £537,833. Gross profit = £37,833 — fleet essentially breaking even. Lost revenue = £302,820. O&M consumes 93% of revenue.",
        "4_climate_risk_tcfd": "Physical climate risk is rated HIGH. June demonstrates the peak summer low-wind scenario. Under 3°C warming, such conditions are projected to become the norm for June–August.",
        "5_strategic_alignment": "June is the strongest argument for SSE's offshore and solar diversification strategy — offshore wind and solar generation peak precisely when onshore wind is weakest.",
        "6_recommended_actions": "1. SUMMER LOW-WIND HEDGING: Evaluate battery storage or demand-response contracts.\n2. SOLAR INTEGRATION ACCELERATION: Wind-solar complementarity directly addresses summer gap.\n3. TCFD SCENARIO UPDATE: Feed June actuals into physical risk models.",
      },
    },
  },
];

export const CHAT_HISTORY = [
  {
    role: "assistant",
    text: "Hello! I'm the SSE Renewables AI Analyst. I have access to live BDH model data and SSE's corporate documents. Ask me anything about wind performance, financial risk, climate exposure, or strategic alignment.",
  },
];

export const MOCK_RESPONSES = {
  default: {
    answer:
      "Based on the BDH model data and SSE's TCFD disclosures, the main physical climate risks are: **Low-wind exposure** (14.2% calm fraction in June, £302,820 lost revenue), **anticyclonic blocking** (increased frequency under 2°C warming pathways per TCFD 2023), and **financial stress** during wind droughts where O&M costs can consume 93% of CfD revenue. Transition risk remains LOW given stable CfD contracts.",
    sources: [
      { file: "SSE_TCFD_2023.pdf", year: "2023", page: 51 },
      { file: "SSE_Annual_Report_2023.pdf", year: "2023", page: 88 },
    ],
  },
  revenue: {
    answer:
      "Revenue analysis across the sample period (Jan–Jun 2023): January was the strongest month at **£2,081,696** CfD revenue (9.24 m/s avg wind), while June was the weakest at **£537,833** (6.51 m/s). The fleet is CfD-protected at £98/MWh, so revenue is purely volumetric — driven entirely by wind resource. Total gross profit across 6 months is approximately **£4.8M** against £3M fixed O&M costs.",
    sources: [{ file: "SSE_Annual_Report_2023.pdf", year: "2023", page: 42 }],
  },
  risk: {
    answer:
      "Risk summary for the 6-month sample period:\n\n**HIGH risk months:** February, April, June — all characterised by anticyclonic blocking, calm fractions above 9%, and BDH memory norm below 0.65.\n\n**MEDIUM risk months:** January, March, May — above-average wind resource, stable BDH memory norm (0.68–0.78), no storm events near cut-out thresholds.\n\nThe primary risk driver is **summer anticyclonic dominance** — June's near-breakeven economics (£37,833 gross profit) represent the most acute realisation of SSE's disclosed low-wind physical risk scenario.",
    sources: [
      { file: "SSE_TCFD_2023.pdf", year: "2023", page: 17 },
      { file: "SSE_TCFD_2023.pdf", year: "2023", page: 51 },
    ],
  },
};

export const KEY_FEATURES = [
  "WS50M", "WS10M", "wind_power_50m", "T2M", "RH2M",
  "PS", "wind_shear_ratio", "memory_norm_last_hour", "CLOUD_AMT", "ALLSKY_SFC_SW_DWN",
];

export const SECTION_KEYS = [
  "1_weather_and_wind_summary",
  "2_energy_generation",
  "3_financial_performance",
  "4_climate_risk_tcfd",
  "5_strategic_alignment",
  "6_recommended_actions",
];

export const SECTION_LABELS = [
  "Weather & Wind",
  "Energy Gen",
  "Financials",
  "Climate Risk",
  "Strategy",
  "Actions",
];