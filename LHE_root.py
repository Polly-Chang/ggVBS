import ROOT
import numpy as np
from lhereader_v2 import LHEReader
from array import array

lhefile = ["../../VBS_task_mg265UL/ggVBS_EW_2134_100000/cmsgrid_final.lhe"]
outfile = "test_EW_mg265UL_0912.root"

reader = []
for file in lhefile:
    reader.append(LHEReader(file))
file = ROOT.TFile.Open(outfile, "RECREATE")
tree = ROOT.TTree("tree", "tree")

jetPID_1 = array("f", [ 1.5 ])
jetPID_2 = array("f", [ 1.5 ])
jetPID_3 = array("f", [ 1.5 ])
gen_weight = array("f", [ 1.5 ])

jet1_ = ROOT.TLorentzVector()
jet2_ = ROOT.TLorentzVector()
jet3_ = ROOT.TLorentzVector()
pho1_ = ROOT.TLorentzVector()
pho2_ = ROOT.TLorentzVector()

tree.Branch("jetPID_1", jetPID_1, "floatb/F")
tree.Branch("jet1_TLor", jet1_)
tree.Branch("jetPID_2", jetPID_2, "floatb/F")
tree.Branch("jet2_TLor", jet2_)
tree.Branch("jetPID_3", jetPID_3, "floatb/F")
tree.Branch("jet3_TLor", jet3_)
tree.Branch("pho1_TLor", pho1_)
tree.Branch("pho2_TLor", pho2_)
tree.Branch("gen_weight", gen_weight, "floatb/F")

# def 

for j in range(len(reader)):
    for iev, event in enumerate(reader[j]):
        if iev == 100000:
            break
        count_jet = 0
        count_pho = 0
        jetPID_1[0] = 0
        jetPID_2[0] = 0
        jetPID_3[0] = 0
        jet1_.SetXYZM(0, 0, 0, 0)
        jet2_.SetXYZM(0, 0, 0, 0)
        jet3_.SetXYZM(0, 0, 0, 0)
        pho1_.SetXYZM(0, 0, 0, 0)
        pho2_.SetXYZM(0, 0, 0, 0)

        gen_weight[0] = event.weights
        num_part = event.num_part

        for i in event.particles:
            isFGluon = (abs(i.pdgid) == 21) and (i.status == 1)
            isFQuark = (1 <= abs(i.pdgid) <= 6) and (i.status == 1)
            isPhoton = (abs(i.pdgid) == 22 and (i.status == 1))

            if ((isFGluon or isFQuark) and count_jet == 0):
                count_jet += 1
                jetPID_1[0] = abs(i.pdgid)
                jetM_1 = i.p4().m
                jetX_1 = i.p4().x
                jetY_1 = i.p4().y
                jetZ_1 = i.p4().z
                jet1_.SetXYZM(jetX_1, jetY_1, jetZ_1, jetM_1)

            elif ((isFGluon or isFQuark) and count_jet == 1):
                count_jet += 1
                jetPID_2[0] = abs(i.pdgid)
                jetM_2 = i.p4().m
                jetX_2 = i.p4().x
                jetY_2 = i.p4().y
                jetZ_2 = i.p4().z
                jet2_.SetXYZM(jetX_2, jetY_2, jetZ_2, jetM_2)

            elif ((isFGluon or isFQuark) and count_jet == 2):
                count_jet += 1
                jetPID_3[0] = abs(i.pdgid)
                jetM_3 = i.p4().m
                jetX_3 = i.p4().x
                jetY_3 = i.p4().y
                jetZ_3 = i.p4().z
                jet3_.SetXYZM(jetX_3, jetY_3, jetZ_3, jetM_3)

            if (isPhoton and count_pho == 0):
                count_pho += 1
                phoM_1 = i.p4().m
                phoX_1 = i.p4().x
                phoY_1 = i.p4().y
                phoZ_1 = i.p4().z
                pho1_.SetXYZM(phoX_1, phoY_1, phoZ_1, phoM_1)
            
            elif (isPhoton and count_pho == 1):
                phoM_2 = i.p4().m
                phoX_2 = i.p4().x
                phoY_2 = i.p4().y
                phoZ_2 = i.p4().z
                pho2_.SetXYZM(phoX_2, phoY_2, phoZ_2, phoM_2)
        tree.Fill()

file.Write()
file.Close()

