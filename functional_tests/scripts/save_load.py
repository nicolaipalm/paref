from functional_tests.blackbox_functions.zdt1 import ZDT1

bbf = ZDT1(input_dimensions=3)
bbf.perform_lhc(20)
print(bbf.evaluations)
evals = bbf.evaluations
bbf.save('test')
bbf.clear_evaluations()
print(bbf.evaluations)

bbf.load('test.npy')
print(bbf.evaluations, bbf.x, bbf.y)
