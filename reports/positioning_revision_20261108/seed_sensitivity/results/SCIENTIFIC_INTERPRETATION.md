# Post-hoc optimizer-seed sensitivity

This analysis was added after the C/D outcomes were known. It uses the existing five paired optimizer seeds for every one of the16 contrasts on each catalog. No fitting, inference, event bootstrap or result selection was repeated. Both catalogs evaluate the same trained models and remain separate; they do not supply ten optimizer seeds.

## Main reporting decision

Use the full seed mean/SD/range/sign table and all five leave-one-seed-out means as the default. The leave-one-out range describes omission sensitivity, not a confidence interval. A stable omission sign means that removing any single observed seed leaves the observed mean sign unchanged; it does not prove a population effect or robustness to different training data.

Optional Student-t intervals are supplied in a separate file only. They require n=5 independent paired optimizer-seed draws and an approximately normal contrast distribution conditional on this source dataset, validation rule and catalog. These assumptions cannot be convincingly checked with five values. The intervals are unadjusted, exploratory, exclude dataset/simulator variation, and must not replace the registered fixed-model event intervals or be described as total training uncertainty.

## Every information-addition contrast

| Catalog | Contrast | Seed mean | Seed SD | + / - / 0 | LOO mean range | LOO direction |
|---|---|---:|---:|---|---|---|
| C | gain_change:Coarse2_minus_Hidden5/f010 | +0.004127366 | 0.000767295 | 5/0/0 | [+0.003904547, +0.004343760] | positive |
| C | gain_change:Coarse8_minus_Coarse2/f010 | -0.007317024 | 0.002210166 | 0/5/0 | [-0.008100387, -0.006798783] | negative |
| C | gain_change:Full5_minus_Coarse8/f010 | -0.001237058 | 0.002136545 | 1/4/0 | [-0.001957663, -0.000546547] | negative |
| C | gain_change:Full5_minus_Hidden5/f010 | -0.004426716 | 0.001795801 | 0/5/0 | [-0.004851898, -0.003681617] | negative |
| C | gain_change:Coarse2_minus_Hidden5/f100 | +0.004661853 | 0.000809121 | 5/0/0 | [+0.004374959, +0.004887936] | positive |
| C | gain_change:Coarse8_minus_Coarse2/f100 | -0.007907531 | 0.000441412 | 0/5/0 | [-0.008023670, -0.007765959] | negative |
| C | gain_change:Full5_minus_Coarse8/f100 | -0.001437462 | 0.000673411 | 0/5/0 | [-0.001638934, -0.001242340] | negative |
| C | gain_change:Full5_minus_Hidden5/f100 | -0.004683141 | 0.001226339 | 0/5/0 | [-0.005029648, -0.004189517] | negative |
| D | gain_change:Coarse2_minus_Hidden5/f010 | +0.004221940 | 0.001046450 | 5/0/0 | [+0.003910172, +0.004452191] | positive |
| D | gain_change:Coarse8_minus_Coarse2/f010 | -0.007302295 | 0.002233169 | 0/5/0 | [-0.008155913, -0.006788945] | negative |
| D | gain_change:Full5_minus_Coarse8/f010 | -0.001444682 | 0.002087145 | 1/4/0 | [-0.002119705, -0.000723095] | negative |
| D | gain_change:Full5_minus_Hidden5/f010 | -0.004525037 | 0.001712870 | 0/5/0 | [-0.004998478, -0.003851103] | negative |
| D | gain_change:Coarse2_minus_Hidden5/f100 | +0.004653161 | 0.000836809 | 5/0/0 | [+0.004367997, +0.004868954] | positive |
| D | gain_change:Coarse8_minus_Coarse2/f100 | -0.007863832 | 0.000526020 | 0/5/0 | [-0.007993305, -0.007692474] | negative |
| D | gain_change:Full5_minus_Coarse8/f100 | -0.001672999 | 0.000805145 | 0/5/0 | [-0.001909415, -0.001415178] | negative |
| D | gain_change:Full5_minus_Hidden5/f100 | -0.004883671 | 0.001446988 | 0/5/0 | [-0.005209283, -0.004271089] | negative |

The main rise at Coarse2-minus-Hidden5 and subsequent fall at Coarse8-minus-Coarse2 must be distinguished from the final step. Report the signs in the table, including any final-step or direct-gain seed exceptions, rather than treating a five-seed mean as unanimous optimizer support.

All32 summary rows,160 original paired seed values and160 omission means are saved in CSV. The TeX main table includes every contrast for each catalog; the optional t table is deliberately separate. None of the earlier manifests or frozen analysis files were rewritten.
