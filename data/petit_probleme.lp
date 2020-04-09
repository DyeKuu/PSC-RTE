\* SP_1 *\
Minimize
OBJ: 5 flot_0,_6 + flot_1,_4 + 3 flot_1,_5 + 4 flot_1,_6
 + 7 flot_1,_9 + 5 flot_2,_4 + 5 flot_2,_6 + 3 flot_2,_7
 + 8 flot_3,_4 + 9 flot_3,_5 + 2 flot_3,_8 + 7 flot_4,_1
 + 4 flot_4,_2 + flot_4,_3 + 2 flot_5,_1 + 3 flot_5,_3 + 3 flot_6,_0
 + 8 flot_6,_1 + 9 flot_6,_2 + 6 flot_7,_2 + flot_8,_3 + 9 flot_9,_1
 + 1000 penal_flot_0,_6 + 1000 penal_flot_1,_4 + 1000 penal_flot_1,_5
 + 1000 penal_flot_1,_6 + 1000 penal_flot_1,_9 + 1000 penal_flot_2,_4
 + 1000 penal_flot_2,_6 + 1000 penal_flot_2,_7 + 1000 penal_flot_3,_4
 + 1000 penal_flot_3,_5 + 1000 penal_flot_3,_8 + 1000 penal_flot_4,_1
 + 1000 penal_flot_4,_2 + 1000 penal_flot_4,_3 + 1000 penal_flot_5,_1
 + 1000 penal_flot_5,_3 + 1000 penal_flot_6,_0 + 1000 penal_flot_6,_1
 + 1000 penal_flot_6,_2 + 1000 penal_flot_7,_2 + 1000 penal_flot_8,_3
 + 1000 penal_flot_9,_1 + 1000 penal_prod_9 + 2 prod_9
Subject To
FlotMax_0,6: flot_0,_6 - penal_flot_0,_6 <= 1
FlotMax_1,4: flot_1,_4 - penal_flot_1,_4 <= 1
FlotMax_1,5: flot_1,_5 - penal_flot_1,_5 <= 1
FlotMax_1,6: flot_1,_6 - penal_flot_1,_6 <= 1
FlotMax_1,9: flot_1,_9 - penal_flot_1,_9 <= 1
FlotMax_2,4: flot_2,_4 - penal_flot_2,_4 <= 1
FlotMax_2,6: flot_2,_6 - penal_flot_2,_6 <= 1
FlotMax_2,7: flot_2,_7 - penal_flot_2,_7 <= 1
FlotMax_3,4: flot_3,_4 - penal_flot_3,_4 <= 1
FlotMax_3,5: flot_3,_5 - penal_flot_3,_5 <= 1
FlotMax_3,8: flot_3,_8 - penal_flot_3,_8 <= 1
FlotMax_4,1: flot_4,_1 - penal_flot_4,_1 <= 1
FlotMax_4,2: flot_4,_2 - penal_flot_4,_2 <= 1
FlotMax_4,3: flot_4,_3 - penal_flot_4,_3 <= 1
FlotMax_5,1: flot_5,_1 - penal_flot_5,_1 <= 1
FlotMax_5,3: flot_5,_3 - penal_flot_5,_3 <= 1
FlotMax_6,0: flot_6,_0 - penal_flot_6,_0 <= 1
FlotMax_6,1: flot_6,_1 - penal_flot_6,_1 <= 1
FlotMax_6,2: flot_6,_2 - penal_flot_6,_2 <= 1
FlotMax_7,2: flot_7,_2 - penal_flot_7,_2 <= 1
FlotMax_8,3: flot_8,_3 - penal_flot_8,_3 <= 1
FlotMax_9,1: flot_9,_1 - penal_flot_9,_1 <= 1

ProdMax_9: - penal_prod_9 + prod_9 <= 15

demand_0: - flot_0,_6 + flot_6,_0 = 9
demand_7: flot_2,_7 - flot_7,_2 = 3
demand_8: flot_3,_8 - flot_8,_3 = 7

flot_1: - flot_1,_4 - flot_1,_5 - flot_1,_6 - flot_1,_9 + flot_4,_1 + flot_5,_1 + flot_6,_1 + flot_9,_1 = 0
flot_2: - flot_2,_4 - flot_2,_6 - flot_2,_7 + flot_4,_2 + flot_6,_2 + flot_7,_2 = 0
flot_3: - flot_3,_4 - flot_3,_5 - flot_3,_8 + flot_4,_3 + flot_5,_3 + flot_8,_3 = 0
flot_4: flot_1,_4 + flot_2,_4 + flot_3,_4 - flot_4,_1 - flot_4,_2 - flot_4,_3 = 0
flot_5: flot_1,_5 + flot_3,_5 - flot_5,_1 - flot_5,_3 = 0
flot_6: flot_0,_6 + flot_1,_6 + flot_2,_6 - flot_6,_0 - flot_6,_1 - flot_6,_2 = 0
flot_9: flot_1,_9 - flot_9,_1 + prod_9 = 0

End
