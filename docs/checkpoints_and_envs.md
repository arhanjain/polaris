# Polaris Checkpoints
All checkpoints for PolaRiS were based on DROID base policies. Checkpoints were produced by cotraining at a weightage of 10% random simulated data and 90% DROID data for 1k steps.

| Policy Name | Checkpoints Path |
| :--- | :--- |
| **π0.5 Polaris** | `gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris` |
| **π0 Fast Polaris** | `gs://openpi-assets/checkpoints/polaris/pi0_fast_droid_jointpos_polaris` |
| **π0 Polaris** | `gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris` |
| **π0 Polaris (100k)** | `gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_100k_polaris` |
| **PaliGemma Polaris** | `gs://openpi-assets/checkpoints/polaris/paligemma_binning_droid_jointpos_polaris` |

## Base DROID Joint Position Checkpoints
| Policy Name | Checkpoints Path |
| :--- | :--- |
| **π0.5 Base** | `gs://openpi-assets/checkpoints/pi05_droid_jointpos` |
| **π0 Fast Base** | `gs://openpi-assets/checkpoints/pi0_fast_droid_jointpos` |
| **π0 Base** | `gs://openpi-assets/checkpoints/pi0_droid_jointpos` |
| **π0 Base (100k)** | `gs://openpi-assets/checkpoints/pi0_droid_jointpos_100k` |
| **PaliGemma Base** | `gs://openpi-assets/checkpoints/paligemma_binning_droid_jointpos` |


# Environments
| Environment Name | Prompt |
| :--- | :--- |
| DROID-BlockStackKitchen | Place and stack the blocks on top of the green tray |
| DROID-FoodBussing | Put all the foods in the bowl |
| DROID-PanClean | Use the yellow sponge to scrub the blue handle frying pan |
| DROID-MoveLatteCup | put the latte art cup on top of the cutting board |
| DROID-OrganizeTools | put the scissor into the large container |
| DROID-TapeIntoContainer | put the tape into the container |
