import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==========================================
# PART 1: SD Model Class Definition
# ==========================================
class OccupationSDModel:
    def __init__(self, params, hist_data):
        self.params = params
        self.hist = hist_data
        self.dt = params['dt']
        self.time_steps = int(params['simulation_years'] / self.dt)
        self.time = np.linspace(2024, 2024 + params['simulation_years'], self.time_steps)
        
        self._process_historical_data()
        
        self.stocks = {
            'Students': np.zeros(self.time_steps),
            'Junior': np.zeros(self.time_steps),
            'Senior': np.zeros(self.time_steps),
            'Unemployed': np.zeros(self.time_steps)
        }
        
        self.stocks['Students'][0] = params['init_students']
        self.stocks['Junior'][0] = params['init_junior']
        self.stocks['Senior'][0] = params['init_senior']
        self.stocks['Unemployed'][0] = 0
        
        self.history = {'Enrollment_Demand': [], 'ROI_Value': []}

    def _process_historical_data(self):
        # 1. Calculate the growth rate
        occ_sal = np.array(self.hist['occupation_salary'])
        years_occ = np.arange(len(occ_sal)).reshape(-1, 1)
        model_occ = LinearRegression().fit(years_occ, np.log(occ_sal))
        self.occ_growth_rate = model_occ.coef_[0]
        
        # 2.Application Premium for Top Universities' Starting Salaries (School Premium)
        school_salary_premium = self.params.get('school_salary_premium', 1.0)
        self.base_occ_salary_2024 = occ_sal[-1] * school_salary_premium
        
        soc_sal = np.array(self.hist['social_salary'])
        years_soc = np.arange(len(soc_sal)).reshape(-1, 1)
        model_soc = LinearRegression().fit(years_soc, np.log(soc_sal))
        self.soc_growth_rate = model_soc.coef_[0]
        self.base_soc_salary_2024 = soc_sal[-1]
        
        # 3. Initial ROI Calculation 
        annual_tuition = self.params['tuition_fee']
        # Net Income = Starting Salary at Prestigious Universities - Tuition Amortization
        net_income = self.base_occ_salary_2024 - (annual_tuition / 10)
        self.initial_roi_index = net_income / self.base_soc_salary_2024

    def sigmoid(self, t, t0, speed):
        return 1 / (1 + np.exp(-speed * (t - t0)))

    def run(self):
        p = self.params
        initial_emp = self.stocks['Junior'][0] + self.stocks['Senior'][0]
        if initial_emp == 0: initial_emp = 1
        C_logistic = (p['K'] / initial_emp) - 1 

        for t in range(self.time_steps - 1):
            curr_year = self.time[t]
            delta_year = curr_year - 2024
            
            S, J, E, U_cum = self.stocks['Students'][t], self.stocks['Junior'][t], self.stocks['Senior'][t], self.stocks['Unemployed'][t]
            
            # 1. Salary Trends
            curr_soc_salary = self.base_soc_salary_2024 * np.exp(self.soc_growth_rate * delta_year)
            natural_occ_salary = self.base_occ_salary_2024 * np.exp(self.occ_growth_rate * delta_year)
            
            # 2. AI and Demand
            ai_maturity = self.sigmoid(curr_year, p['ai_boom_year'], p['ai_speed'])
            natural_demand = p['K'] / (1 + C_logistic * np.exp(-p['r'] * delta_year))
            real_demand = natural_demand * (1 - p['alpha'] * ai_maturity * 0.25)
            
            gap = real_demand - (J + E)
            gap_ratio = gap / (real_demand + 1e-9)
            
            # 3. Salary Feedback and ROI
            salary_gap_effect = np.clip(1 + (gap_ratio * p['salary_sensitivity']), 0.7, 1.5)
            actual_occ_salary = natural_occ_salary * salary_gap_effect
            
            current_tuition = p['tuition_fee'] * (1.03 ** delta_year) 
            net_income_expected = actual_occ_salary - (current_tuition / 10)
            current_roi_index = net_income_expected / curr_soc_salary
            
            roi_ratio = current_roi_index / (self.initial_roi_index + 1e-5)
            attractiveness_factor = (roi_ratio ** p['enrollment_elasticity']) * p['brand_premium']
            
            # 4. Enrollment Demand
            enrollment_demand = p['base_enrollment'] * attractiveness_factor
            
            # Actual enrollment 
            max_cap = p.get('max_school_capacity', p['base_enrollment'] * 3)
            actual_enrollment = min(enrollment_demand, max_cap)
            
            # AI Cost Pressure
            current_pure_premium = actual_occ_salary / curr_soc_salary
            cost_push = 1 + (current_pure_premium - 1.2) * p['cost_push_factor']
            ai_adoption_pressure = ai_maturity * max(0.5, cost_push)
            
            grad_join = S / p['school_years']
            promotion = J * p['promotion_rate']
            j_exit_natural = J * p['natural_turnover']
            e_exit_natural = E * p['retirement_rate']
            j_ai_replace = J * p['alpha'] * ai_adoption_pressure * p['vuln_junior']
            e_ai_replace = E * p['alpha'] * ai_adoption_pressure * p['vuln_senior']
            
            dS = actual_enrollment - grad_join
            dJ = grad_join - promotion - j_exit_natural - j_ai_replace
            dE = promotion - e_exit_natural - e_ai_replace
            dU = j_ai_replace + e_ai_replace
            
            self.stocks['Students'][t+1] = S + dS * self.dt
            self.stocks['Junior'][t+1] = J + dJ * self.dt
            self.stocks['Senior'][t+1] = E + dE * self.dt
            self.stocks['Unemployed'][t+1] = U_cum + dU * self.dt
            
            self.history['Enrollment_Demand'].append(enrollment_demand)
            self.history['ROI_Value'].append(current_roi_index)
            
        for k in self.history:
            self.history[k].append(self.history[k][-1])

# ==========================================
# PART 2: Parameter Configuration for Specific Institutions
# ==========================================
social_history_data = [56310, 58260, 61900, 65470, 67920]

configs = [
    {
        'legend': 'RIT (Programmer): Grow',
        'color': '#1f77b4', 
        'action_text': 'Grow',
        'salary_data': [61650, 62235, 63980, 65497.5, 66965, 68642.5, 70115, 72217.5, 73575, 76220, 78335, 80307.5, 85627.5, 89775, 91167.5],
        'params': {
            'school_name': 'RIT',
            'tuition_fee': 54518, 'init_students': 13152, 'base_enrollment': 3056, 'school_years': 4,
            'school_salary_premium': 1.2, 
            'dt': 0.1, 'simulation_years': 16, 
            'init_junior': 1800000, 'init_senior': 500000,
            'r': 0.0230, 'K': 3811230, 'alpha': 0.5224,
            'promotion_rate': 0.15, 'natural_turnover': 0.15, 'retirement_rate': 0.05,
            'ai_boom_year': 2028, 'ai_speed': 0.8,
            'vuln_junior': 0.25, 'vuln_senior': 0.05,
            'salary_sensitivity': 1.5, 'enrollment_elasticity': 2.0, 'cost_push_factor': 1.2,
            'brand_premium': 1.2
        }
    },
    {
        'legend': 'CIA (Chef): Maintain',
        'color': '#ff7f0e', 
        'action_text': 'Maintain',
        'salary_data': [26784, 27187, 25725, 25872, 26031, 27746, 29186, 30286, 31122, 32922, 33855, 34362, 36772, 39341, 40806],
        'params': {
            'school_name': 'CIA',
            'tuition_fee': 36620, 'init_students': 3005, 'base_enrollment': 680, 'school_years': 4,
            'school_salary_premium': 1.8, 
            'dt': 0.1, 'simulation_years': 16,
            'init_junior': 5000000, 'init_senior': 1000000,
            'r': 0.0183, 'K': 13952490, 'alpha': 0.4367,
            'promotion_rate': 0.10, 'natural_turnover': 0.15, 'retirement_rate': 0.08,
            'ai_boom_year': 2030, 'ai_speed': 0.4,
            'vuln_junior': 0.15, 'vuln_senior': 0.02,
            'salary_sensitivity': 0.8, 'enrollment_elasticity': 2.5, 'cost_push_factor': 0.5,
            'brand_premium': 1.5 
        }
    },
    {
        'legend': 'Berklee (Singer): Shrink',
        'color': '#d62728', 
        'action_text': 'Shrink',
        'salary_data': [62857.6, 66019.2, 66435.2, 66768, 67579.2, 69929.6, 71884.8, 74588.8, 78020.8, 83116.8, 86777.6, 86236.8, 104436.8, 107432, 114670.4],
        'params': {
            'school_name': 'Berklee',
            'tuition_fee': 48330, 'init_students': 7395, 'base_enrollment': 1502, 'school_years': 4,
            'school_salary_premium': 1.1,
            'dt': 0.1, 'simulation_years': 16,
            'init_junior': 35000, 'init_senior': 8000,
            'r': -0.0189, 'K': 46310, 'alpha': 0.3411,
            'promotion_rate': 0.05, 'natural_turnover': 0.10, 'retirement_rate': 0.04,
            'ai_boom_year': 2026, 'ai_speed': 0.9,
            'vuln_junior': 0.20, 'vuln_senior': 0.30,
            'salary_sensitivity': 1.2, 'enrollment_elasticity': 1.5, 'cost_push_factor': 0.8,
            'brand_premium': 1.3
        }
    }
]

# ==========================================
# PART 3: Run and Plotting
# ==========================================

results = {}
for config in configs:
    hist_pkg = {'occupation_salary': config['salary_data'], 'social_salary': social_history_data}
    model = OccupationSDModel(config['params'], hist_pkg)
    model.run()
    
    base_demand = model.history['Enrollment_Demand'][0]
    if base_demand == 0: base_demand = 1
    norm_demand = np.array(model.history['Enrollment_Demand']) / base_demand * 100
    
    results[config['params']['school_name']] = {
        'time': model.time,
        'demand_norm': norm_demand,
        'roi': model.history['ROI_Value'],
        'legend': config['legend'],
        'color': config['color'],
        'action': config['action_text']
    }

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
years = results['RIT']['time']

# --- Subplot 1: Enrollment ---
#AI Statement: This section utilized Gemini 3 Pro to adjust the positioning of the labels in the image to resolve overlapping text issues.
for name, data in results.items():
    ax1.plot(data['time'], data['demand_norm'], color=data['color'], linewidth=2.5, label=data['legend'])
    y_final = data['demand_norm'][-1]
    y_offset = 0
    
    if name == 'RIT': 
        y_offset = -5   
    elif name == 'CIA': 
        y_offset = 18   
    elif name == 'Berklee': 
        y_offset = -18 
    
    ax1.text(2039.5, y_final + y_offset, data['action'], 
             color=data['color'], fontweight='bold', va='center', ha='right', fontsize=11)

ax1.axhline(100, color='gray', linewidth=1, alpha=0.5, linestyle='--')
ax1.set_ylabel('Enrollment Demand\n(% of 2024 Base)', fontsize=12)
ax1.set_title('Strategic Comparison: Enrollment Recommendations', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# --- Subplot 2: ROI ---
for name, data in results.items():
    ax2.plot(data['time'], data['roi'], color=data['color'], linewidth=2, label=f"{name} ROI")

ax2.axhline(1.0, color='black', linewidth=1.5, linestyle='-', label='Break-even (ROI=1)')
ax2.fill_between(years, 0, 1.0, color='red', alpha=0.1)
ax2.text(2032, 0.75, 'Financial Danger Zone (ROI < 1)', 
         color='red', alpha=0.5, fontsize=12, ha='center', va='center')

ax2.set_ylabel('ROI Index\n(Net Income / Avg Salary)', fontsize=12)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_title('Economic Rationale: Education ROI Trends', fontsize=14, fontweight='bold')
ax2.legend(loc='lower left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.5, 3.0) 

plt.tight_layout()
plt.show()
