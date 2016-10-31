from bayesian.factor_graph import *
from bayesian.bbn import build_bbn

def f_age(age):
    """https://en.wikipedia.org/wiki/Demography_of_the_United_States"""
    if age == 'Low':
        return 0.35
    elif age == 'Medium':
        return 0.35
    else:
        return 0.30


def f_education(education):
    """https://en.wikipedia.org/wiki/Educational_attainment_in_the_United_States"""
    if education == 'Low':
        return 0.15
    elif education == 'Medium':
        return 0.55
    else:
        return 0.30

def f_sex(sex):
    """https://en.wikipedia.org/wiki/Demography_of_the_United_States"""
    """They're roughly the same"""
    return 0.5

def f_job(job):
    """https://www.google.com/webhp?sourceid=chrome-instant&ion=1&espv=2&ie=UTF-8#q=us%20unemployment%20rate"""
    if(job):
        return 0.951
    else:
        return 0.049

def f_us_part(us_part):
    """This is a veeeeery vague thing, so 50/50"""
    if us_part == 'North':
        return 0.5
    elif us_part == 'South':
        return 0.5

def f_socialization(socialization):
    """Also vague"""
    if (socialization == 'High'):
        return 0.9
    else:
        return 0.1
        
def f_marriage(age, marriage):
    """https://en.wikipedia.org/wiki/Marriage_in_the_United_States"""
    table = dict()
    table['Low'] = 0.4
    table['Medium'] = 0.65
    table['High'] = 0.5
    if (marriage):
        return table[age]
    else:
        return 1 - table[age]
        
def f_race(race):
    """http://www.forumbiodiversity.com/showthread.php/9181-World-population-by-race"""
    if race == 'Caucasian':
        return 0.80
    elif race == 'Asian':
        return 0.07
    elif race == 'Black':
        return 0.07
    elif race == 'Australoid':
        return 0.06

def f_business(age, education, sex, business):
    table = dict()
    """P('Has Business')"""
    table['LLF'] = 0.01
    table['LLM'] = 0.02
    table['LMF'] = 0.02
    table['LMM'] = 0.03
    table['LHF'] = 0.04
    table['LHM'] = 0.05
    
    table['MLF'] = 0.02
    table['MLM'] = 0.03
    table['MMF'] = 0.04
    table['MMM'] = 0.05
    table['MHF'] = 0.05
    table['MHM'] = 0.07
    
    table['HLF'] = 0.03
    table['HLM'] = 0.04
    table['HMF'] = 0.03
    table['HMM'] = 0.05
    table['HHF'] = 0.03
    table['HHM'] = 0.06
    key = ''
    key += age[0]
    key += education[0]
    key += sex[0]
    if(business):
        return table[key]
    else:
        return 1 - table[key]

def f_migrant(race, migrant):
    table = dict()
    """http://www.migrationpolicy.org/data/state-profiles/state/demographics/US"""
    """P('Migrant')"""
    table['Caucasian'] = 0.03
    table['Asian'] = 0.05
    table['Black'] = 0.01
    table['Australoid'] = 0.01
    if (migrant):
        return table[race]
    else:
        return 1 - table[race]

def f_income(business, job, income):
    table = dict()
    """P('Level of Income')"""
    table['ttL'] = 0.05
    table['ttM'] = 0.75
    table['ttH'] = 0.20
    
    table['tfL'] = 0.05
    table['tfM'] = 0.60
    table['tfH'] = 0.35
    
    table['ftL'] = 0.20
    table['ftM'] = 0.75
    table['ftH'] = 0.05
    
    table['ffL'] = 0.950
    table['ffM'] = 0.049
    table['ffH'] = 0.001
    key = ''
    key = key + 't' if business else key + 'f'
    key = key + 't' if job else key + 'f'
    key += income[0]
    return table[key]


def f_views(us_part, marriage, income, views):
    table = dict()
    """P('Conservative')"""
    table['StH'] = 0.8
    table['StM'] = 0.6
    table['StL'] = 0.4
    table['SfH'] = 0.7
    table['SfM'] = 0.5
    table['SfL'] = 0.4
    
    table['NtH'] = 0.6
    table['NtM'] = 0.4
    table['NtL'] = 0.3
    table['NfH'] = 0.5
    table['NfM'] = 0.4
    table['NfL'] = 0.2
    key = ''
    key += us_part[0]
    key = key + 't' if marriage else key + 'f'
    key += income[0]
    if(views == 'Conservative'):
        return table[key]
    else:
        return 1 - table[key]

def f_for_policy(income, migrant, for_policy):
    table = dict()
    """P('To have Aggressive foreign policy veiws')"""
    table['tL'] = 0.05
    table['tM'] = 0.10
    table['tH'] = 0.20
    table['fL'] = 0.60
    table['fM'] = 0.20
    table['fH'] = 0.50
    key = ''
    key = key + 't' if migrant else key + 'f'
    key += income[0]
    if for_policy == 'Aggressive':
        return table[key]
    else:
        return 1 - table[key]
        
def f_candidate(socialization, views, for_policy, candidate):
    table = dict()
    """P('To vote for Trump')"""
    table['LCA'] = 0.90
    table['LCN'] = 0.60
    table['LLA'] = 0.50
    table['LLN'] = 0.40
    
    table['HCA'] = 0.70
    table['HCN'] = 0.50
    table['HLA'] = 0.50
    table['HLN'] = 0.30
    key = ''
    key += socialization[0]
    key += views[0]
    key += for_policy[0]
    if (candidate == 'Trump'):
        return table[key]
    else:
        return 1 - table[key]
        
if __name__ == '__main__':
    g = build_graph(
        f_age,
        f_education,
        f_sex,
        f_race,
        f_migrant,
        f_business,
        f_job,
        f_income,
        f_marriage,
        f_us_part,
        f_views,
        f_socialization,
        f_candidate,
        f_for_policy,
        domains={
            'race': ['Caucasian', 'Asian', 'Black', 'Australoid'],
            'education': ['Low', 'Medium', 'High'],
            'age': ['Low', 'Medium', 'High'],
            'income': ['Low', 'Medium', 'High'],
            'us_part': ['North', 'South'],
            'sex': ['Male', 'Female'],
            'views': ['Conservative', 'Liberal'],
            'for_policy': ['Aggressive', 'Nonaggressive'],
            'candidate': ['Trump', 'Hillary'],
            'socialization': ['High', 'Low']})
    g.n_samples = 1000
    g.q()
