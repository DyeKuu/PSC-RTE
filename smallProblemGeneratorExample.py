from package_name.generator import small_problem_generator

numGeneratedProblem = 100
numVariedRHS = 10
deviation = 0.1
nameProblem = ["problem-1-1-20190923-142640.mps"]
smallProblemData = small_problem_generator(nameProblem,numGeneratedProblem,numVariedRHS ,deviation)
smallProblemData.dump_in_file("smallProblemN="+str(numGeneratedProblem)+"-m="+str(numVariedRHS)+"-Dev="+str(deviation))
