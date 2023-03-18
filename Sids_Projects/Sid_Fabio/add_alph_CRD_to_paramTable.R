library(data.table)


chosen_CRD <- c(640, 650, 651, 1250, 2680)
chosen_CRD_alph <- c("CA40", "CA50", "CA51", "FL50", "MI80")


param_dir <- "/Users/hn/Documents/01_research_data/Sid/SidFabio/parameters/"
tomato_crd_trial <- data.table(read.csv(paste0(param_dir, "tomato_crd_trial.csv")))


chosen_CRD_tb <- data.table()
chosen_CRD_tb$STASD_N=chosen_CRD
chosen_CRD_tb$CRD=chosen_CRD_alph

tomato_crd_trial <- dplyr::left_join(x=tomato_crd_trial, y=alphabet_CRD, by="STASD_N")

write.csv(tomato_crd_trial, paste0(param_dir, "tomato_crd_trial.csv"))