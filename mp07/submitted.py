'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import copy, queue

def standardize_variables(nonstandard_rules):
    '''
    @param nonstandard_rules (dict) - dict from ruleIDs to rules
        Each rule is a dict:
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        consequent contains the rule consequent (a proposition).
   
    @return standardized_rules (dict) - an exact copy of nonstandard_rules,
        except that the antecedents and consequent of every rule have been changed
        to replace the word "something" with some variable name that is
        unique to the rule, and not shared by any other rule.
    @return variables (list) - a list of the variable names that were created.
        This list should contain only the variables that were used in rules.
    '''
    # raise RuntimeError("You need to write this part!")

    variables = []
    standardized_rules = copy.deepcopy(nonstandard_rules)

    # for rule in standardized_rules:
    #     antecedents = standardized_rules[rule]['antecedents']
    #     print(antecedents)



    for rule_key, rule in standardized_rules.items():
        # print(rule)
        antecedents_list = rule['antecedents']
        # print(antecedents)
        consequent = rule['consequent']
        # print(consequent)
        length = len(variables)
        x_str = 'x' + '0' * (4 - len(str(length))) + str(length)
        # print(standardized_rules)
        flag = 0
        for i, antecedents in enumerate(antecedents_list):
            if 'something' in antecedents:
                flag = 1
                temp = rule['antecedents'][i].index('something')
                rule['antecedents'][i] = rule['antecedents'][temp][:temp]+[x_str]+rule['antecedents'][temp][temp+1:]
                # rule['antecedents'][i] = antecedent.replace('something', variable_name)

        if 'something' in consequent:
            temp = consequent.index('something')
            rule['consequent'] = rule['consequent'][:temp]+[x_str]+rule['consequent'][temp+1:]
            # consequent = consequent.replace('something', variable_name)
        if flag == 1:
          variables.append(x_str)
        
    return standardized_rules, variables

def unify(query, datum, variables):
    '''
    @param query: proposition that you're trying to match.
      The input query should not be modified by this function; consider deepcopy.
    @param datum: proposition against which you're trying to match the query.
      The input datum should not be modified by this function; consider deepcopy.
    @param variables: list of strings that should be considered variables.
      All other strings should be considered constants.
    
    Unification succeeds if (1) every variable x in the unified query is replaced by a 
    variable or constant from datum, which we call subs[x], and (2) for any variable y
    in datum that matches to a constant in query, which we call subs[y], then every 
    instance of y in the unified query should be replaced by subs[y].

    @return unification (list): unified query, or None if unification fails.
    @return subs (dict): mapping from variables to values, or None if unification fails.
       If unification is possible, then answer already has all copies of x replaced by
       subs[x], thus the only reason to return subs is to help the calling function
       to update other rules so that they obey the same substitutions.

    Examples:

    unify(['x', 'eats', 'y', False], ['a', 'eats', 'b', False], ['x','y','a','b'])
      unification = [ 'a', 'eats', 'b', False ]
      subs = { "x":"a", "y":"b" }
    unify(['bobcat','eats','y',True],['a','eats','squirrel',True], ['x','y','a','b'])
      unification = ['bobcat','eats','squirrel',True]
      subs = { 'a':'bobcat', 'y':'squirrel' }
    unify(['x','eats','x',True],['a','eats','a',True],['x','y','a','b'])
      unification = ['a','eats','a',True]
      subs = { 'x':'a' }
    unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'x':'a', 'a':'bobcat'}
      When the 'x':'a' substitution is detected, the query is changed to 
      ['a','eats','a',True].  Then, later, when the 'a':'bobcat' substitution is 
      detected, the query is changed to ['bobcat','eats','bobcat',True], which 
      is the value returned as the answer.
    unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'a':'x', 'x':'bobcat'}
      When the 'a':'x' substitution is detected, the query is changed to 
      ['x','eats','bobcat',True].  Then, later, when the 'x':'bobcat' substitution 
      is detected, the query is changed to ['bobcat','eats','bobcat',True], which is 
      the value returned as the answer.
    unify([...,True],[...,False],[...]) should always return None, None, regardless of the 
      rest of the contents of the query or datum.
    '''
    # raise RuntimeError("You need to write this part!")
    # def unify(query, datum, variables):
    query_ = copy.deepcopy(query)
    datum_ = copy.deepcopy(datum)
    subs = {}
    
    if query_[-1] == True and datum_[-1] == False:
      return None, None
    
    for i in range(len(query_)):
      word1 = query_[i]
      word2 = datum_[i]
      if word1 in variables:
        subs[word1] = word2
        query_ = [subs[word1] if i == word1 else i for i in query_]

      elif word2 in variables:
        subs[word2] = word1
        query_ = [subs[word2] if j == word2 else j for j in query_]

      elif word2 != word1:
        return None, None

    return query_, subs

def apply(rule, goals, variables):
    '''
    @param rule: A rule that is being tested to see if it can be applied
      This function should not modify rule; consider deepcopy.
    @param goals: A list of propositions against which the rule's consequent will be tested
      This function should not modify goals; consider deepcopy.
    @param variables: list of strings that should be treated as variables

    Rule application succeeds if the rule's consequent can be unified with any one of the goals.
    
    @return applications: a list, possibly empty, of the rule applications that
       are possible against the present set of goals.
       Each rule application is a copy of the rule, but with both the antecedents 
       and the consequent modified using the variable substitutions that were
       necessary to unify it to one of the goals. Note that this might require 
       multiple sequential substitutions, e.g., converting ('x','eats','squirrel',False)
       based on subs=={'x':'a', 'a':'bobcat'} yields ('bobcat','eats','squirrel',False).
       The length of the applications list is 0 <= len(applications) <= len(goals).  
       If every one of the goals can be unified with the rule consequent, then 
       len(applications)==len(goals); if none of them can, then len(applications)=0.
    @return goalsets: a list of lists of new goals, where len(newgoals)==len(applications).
       goalsets[i] is a copy of goals (a list) in which the goal that unified with 
       applications[i]['consequent'] has been removed, and replaced by 
       the members of applications[i]['antecedents'].

    Example:
    rule={
      'antecedents':[['x','is','nice',True],['x','is','hungry',False]],
      'consequent':['x','eats','squirrel',False]
    }
    goals=[
      ['bobcat','eats','squirrel',False],
      ['bobcat','visits','squirrel',True],
      ['bald eagle','eats','squirrel',False]
    ]
    variables=['x','y','a','b']

    applications, newgoals = submitted.apply(rule, goals, variables)

    applications==[
      {
        'antecedents':[['bobcat','is','nice',True],['bobcat','is','hungry',False]],
        'consequent':['bobcat','eats','squirrel',False]
      },
      {
        'antecedents':[['bald eagle','is','nice',True],['bald eagle','is','hungry',False]],
        'consequent':['bald eagle','eats','squirrel',False]
      }
    ]
    newgoals==[
      [
        ['bobcat','visits','squirrel',True],
        ['bald eagle','eats','squirrel',False]
        ['bobcat','is','nice',True],
        ['bobcat','is','hungry',False]
      ],[
        ['bobcat','eats','squirrel',False]
        ['bobcat','visits','squirrel',True],
        ['bald eagle','is','nice',True],
        ['bald eagle','is','hungry',False]
      ]
    '''
    # raise RuntimeError("You need to write this part!")

    rule_ = copy.deepcopy(rule)
    goals_ = copy.deepcopy(goals)

    applications = []
    goalsets = []
    for i in range(len(goals_)):
      new_rule = copy.deepcopy(rule_)
      new_goal = copy.deepcopy(goals_)
      consequent = new_rule['consequent']
      antecedent_list = new_rule['antecedents']
      unification, subs = unify(new_goal[i], consequent, variables) #get the unification result

      if unification == None: #ignore the goal that cannot match any rule
         continue
      
      new_rule['consequent'] = unification
      new_goal.remove(new_goal[i])  #remove the goal that match with rule

      for j in range(len(antecedent_list)):
         antecedent = antecedent_list[j]

         for word in antecedent:
            if word in subs:
               antecedent_list[j] = [subs[word] if i == word else i for i in antecedent]  #sub word in antecedent

      applications.append(new_rule) #add changed antecedents and consequent

      for k in range(len(antecedent_list)):
         new_goal = new_goal + [antecedent_list[k]]  #add matched rule to new goal

      goalsets.append(new_goal)

    return applications, goalsets

def backward_chain(query, rules, variables):
    '''
    @param query: a proposition, you want to know if it is true
    @param rules: dict mapping from ruleIDs to rules
    @param variables: list of strings that should be treated as variables

    @return proof (list): a list of rule applications
      that, when read in sequence, conclude by proving the truth of the query.
      If no proof of the query was found, you should return proof=None.
    '''
    # raise RuntimeError("You need to write this part!")

    proof = []
    queue = []
    queue.append(([query], []))

    while(queue):
       goals, proof = queue.pop()
       if len(goals) == 0:
          return proof
       
       for rule_key, rule in rules.items():
          applications, newgoals = apply(rule, goals, variables)
          for i in newgoals:
             queue.append((i, applications + proof))

    return None
