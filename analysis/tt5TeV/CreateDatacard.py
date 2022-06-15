from cafea.modules.CreateDatacardFromRootfile import Datacard
path = '/nfs/fanae/user/juanr/CMSSW_10_2_13/src/tt5TeV/ljets/'
fname = 'NBtags.root'

lumiUnc = 0.015
bkg =  ['tW', 'WJets', 'QCD', 'DY']
norm = [0.2, 0.2, 0.2, 0.2]
signal = 'tt'
systList = ['lepSF', 'trigSF', 'btagSF', 'FSR']#, 'ISR', 'hdamp', 'Tune', 'Scales', 'PDF', 'Prefire']
d = Datacard(path+fname, signal, bkg, lumiUnc, norm, systList, nSpaces=12)
d.AddExtraUnc('prefiring', 0.014, ['tt', 'tW', 'WJets', 'DY'])
d.AddExtraUnc('Tune', 0.007, signal)
d.AddExtraUnc('hdamp', 0.01, signal)
d.AddExtraUnc('PDF', 0.003, signal)
d.AddExtraUnc('Scales', 0.002, signal)
d.SetOutPath(path)
d.Save('datacard.txt')
