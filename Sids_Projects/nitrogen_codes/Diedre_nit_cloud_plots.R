rm(list=ls())
library(data.table)
library(dplyr)
library(ggmap)
library(ggplot2)
library(plotly)
# library(plot3D)
options(digit=9)
options(digits=9)

################################
########
########  Directories
########
################################
data_dir = "/Users/hn/Documents/01_research_data/Sid/Nitrogen_data/"

################################
########
########         Read
########
################################
Deirdre            = read.csv(paste0(data_dir, "Deirdre/03_scaled_Deirdre.csv"))
smoothed_daily     = read.csv(paste0(data_dir, "01_corn_potatoEq2_smoothed_daily.csv"))
# scaled_shifted_nit = read.csv(paste0(data_dir, "03_scaled_shifted_corn_potatoEq2_smoothed.csv"))
yaxis_scaled_shifted_nit = read.csv(paste0(data_dir, "03_scaled_yAxis_corn_potatoEq2_smoothed.csv"))

################################
########
########  Change column names of Deirdre
########
################################
library(stringr)
colnames_Deirdre <- colnames(Deirdre)
new_cols <- str_replace_all(colnames_Deirdre, "\\.\\.", "_")
new_cols <- str_replace_all(new_cols, "\\.", "_")

setnames(Deirdre, old=colnames_Deirdre, new=new_cols)

################################
########
########         Plot Deirdre
########
################################
 # stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                         #              fun.min=function(z) { quantile(z,0) }, 
                         #              fun.max=function(z) { quantile(z,1) }, 
                         #              alpha=0.2) +
                         # stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                         #              fun.min=function(z) { quantile(z,0.1) }, 
                         #              fun.max=function(z) { quantile(z,0.9) }, 
                         #              alpha=0.4) +
                         # stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                         #              fun.min=function(z) { quantile(z,0.25) }, 
                         #              fun.max=function(z) { quantile(z,0.75) }, 
                         #              alpha=0.8) +
axis_label_font_size <- 12
axis_text_font_size <- axis_label_font_size-2
title_font_size <- axis_label_font_size+2

Inorganic_N_g_m2_mean <- ggplot(Deirdre, aes(x=Day, y=Inorganic_N_g_m2_mean, color=factor(Product_Type))) +
                         labs(x = "",y = expression(paste("Inorganic N (g/", m^{2}, ") mean"))) + # , fill = "Product_Type"
                         guides(fill=guide_legend(title="")) + 
                         geom_line() + 
                         # facet_grid(. ~ Product_Type, scales="free") +
                         stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}, size=1) + 
                         #scale_color_manual(values=c("darkgreen", "orange", "red"))+
                         #scale_fill_manual(values=c("darkgreen", "orange", "red"))+
                         theme(panel.grid.major = element_line(size=0.2),
                               panel.spacing=unit(.5, "cm"),
                               legend.text=element_text(size=axis_label_font_size), # face="bold"
                               legend.title = element_blank(),
                               legend.position = "bottom",
                               strip.text = element_text(face="bold", size=axis_text_font_size, color="black"),
                               axis.text = element_text(size=axis_text_font_size, color="black"), # face="bold",
                               axis.text.x = element_blank(),
                               axis.ticks = element_line(color = "black", size = .2),
                               axis.title.x = element_text(margin=margin(t=-1, r=0, b=-1, l=0)),
                               axis.title.y = element_text(size=axis_label_font_size,
                                                           margin=margin(t=0, r=10, b=0, l=0)),
                               plot.title = element_text(lineheight=.8, face="bold", size=title_font_size)
                               )


Mineralized_Inorganic <- ggplot(Deirdre, aes(x=Day, y=Mineralized_Inorganic_N_g_m2_, color=factor(Product_Type))) +
                         labs(x = "Day",y = expression(paste("Mineralized Inorganic N (g/", m^{2}, ")"))) + # , fill = "Product_Type"
                         guides(fill=guide_legend(title="")) + 
                         geom_line() + 
                         stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}, size=1) + 
                         theme(panel.grid.major = element_line(size=0.2),
                               panel.spacing=unit(.5, "cm"),
                               legend.text=element_text(size=axis_label_font_size, face="bold"),
                               legend.title = element_blank(),
                               legend.position = "bottom",
                               strip.text = element_text(face="bold", size=axis_text_font_size, color="black"),
                               axis.text = element_text(size=axis_text_font_size, color="black"), # face="bold",
                               axis.ticks = element_line(color = "black", size = .2),
                               axis.title.x = element_text(size=axis_label_font_size, 
                                                           margin=margin(t=10, r=0, b=0, l=0)),
                               axis.title.y = element_text(size=axis_label_font_size,
                                                           margin=margin(t=0, r=10, b=0, l=0)),
                               plot.title = element_text(lineheight=.8, face="bold", size=title_font_size)
                               )

Deirdre_plots <- ggpubr::ggarrange(plotlist = list(Inorganic_N_g_m2_mean, Mineralized_Inorganic),
                                   ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")

plot_dir <- paste0(data_dir, "Deirdre/")
ggsave(filename = "Deirdre_scaled.png",
       plot = Deirdre_plots, width=5.5, height=7, units = "in", 
       dpi=300, device = "png", path = plot_dir)

################################
########
########   Plot shifted uptake
########
################################

head(scaled_shifted_nit, 2) 
doys <- rep(1:365, 878555/365)
scaled_shifted_nit$day = doys
smoothed_daily$day = doys

 # stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                  #              fun.min=function(z) { quantile(z,0) }, 
                  #              fun.max=function(z) { quantile(z,1) }, 
                  #              alpha=0.2) +

shifted_uptake <- ggplot(scaled_shifted_nit, aes(x=day, y=smooth_window3, color=factor(CropTyp), fill=factor(CropTyp))) + 
                  labs(x = "", y=expression(paste("nitrogen content ", (gm^{-2})))) + # colour="CropTyp"
                  # guides(fill=guide_legend(title="")) + 
                  facet_grid(~ CropTyp, scales="free") +
                  stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}, size=1) + 
                  stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                               fun.min=function(z) { quantile(z,0.1) }, 
                               fun.max=function(z) { quantile(z,0.9) }, 
                               alpha=0.4) + # fill = ("darkgreen")
                  stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                               fun.min=function(z) { quantile(z,0.25) }, 
                               fun.max=function(z) { quantile(z,0.75) }, 
                               alpha=0.8) + # fill = ("orange")
                  # scale_fill_brewer(palette = "Set1") +
                  scale_colour_manual(values=c("darkgreen", "dodgerblue")) +
                  scale_fill_manual(values=c("darkgreen", "dodgerblue")) +
                  ggtitle("shifted and scaled") +
                  theme(panel.grid.major = element_line(size=0.2),
                        panel.spacing=unit(.5, "cm"),
                        legend.text=element_text(size=axis_label_font_size), # face="bold"
                        legend.title = element_blank(),
                        legend.position = "none", # "bottom",
                        strip.text = element_text(face="bold", size=axis_text_font_size, color="black"),
                        axis.text = element_text(size=axis_text_font_size, color="black"), # face="bold",
                        axis.text.x = element_blank(),
                        axis.ticks = element_line(color = "black", size = .2),
                        axis.title.x = element_text(margin=margin(t=-1, r=0, b=-1, l=0)),
                        axis.title.y = element_text(size=axis_label_font_size,
                                                    margin=margin(t=0, r=10, b=0, l=0)),
                        plot.title = element_text(lineheight=.8, face="bold", size=title_font_size)
                        )


noShift_uptake <- ggplot(smoothed_daily, aes(x=day, y=smooth_window3, color=factor(CropTyp), fill=factor(CropTyp))) + 
                  labs(x = "", y=expression(paste("nitrogen content ", (gm^{-2})))) + # colour="CropTyp"
                  # guides(fill=guide_legend(title="")) + 
                  facet_wrap(~ CropTyp, scale="free") +
                  stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}, size=1) + 
                  
                  stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                               fun.min=function(z) { quantile(z,0.1) }, 
                               fun.max=function(z) { quantile(z,0.9) }, 
                               alpha=0.4) +

                  stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                               fun.min=function(z) { quantile(z,0.25) }, 
                               fun.max=function(z) { quantile(z,0.75) }, 
                               alpha=0.8) + 


                  # stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                  #              fun.min=function(z) { quantile(z,0) }, 
                  #              fun.max=function(z) { quantile(z,1) }, 
                  #              alpha=0.2) +

                  # scale_fill_brewer(palette = "Set1") +
                  scale_colour_manual(values=c("darkgreen", "dodgerblue")) +
                  scale_fill_manual(values=c("darkgreen", "dodgerblue")) +
                  ggtitle("actual values and actual DoY") +
                  theme(panel.grid.major = element_line(size=0.2),
                        panel.spacing=unit(.5, "cm"),
                        legend.text=element_text(size=axis_label_font_size), # face="bold"
                        legend.title = element_blank(),
                        legend.position = "none", # "bottom",
                        strip.text = element_text(face="bold", size=axis_text_font_size, color="black"),
                        axis.text = element_text(size=axis_text_font_size, color="black"), # face="bold",
                        axis.text.x = element_blank(),
                        axis.ticks = element_line(color = "black", size = .2),
                        axis.title.x = element_text(margin=margin(t=-1, r=0, b=-1, l=0)),
                        axis.title.y = element_text(size=axis_label_font_size,
                                                    margin=margin(t=0, r=10, b=0, l=0)),
                        plot.title = element_text(lineheight=.8, face="bold", size=title_font_size)
                        )


yaxis_shifted_uptake <- ggplot(yaxis_scaled_shifted_nit, aes(x=x_axis, y=smooth_window3, color=factor(CropTyp), fill=factor(CropTyp))) + 
                        labs(x = "", y=expression(paste("nitrogen content ", (gm^{-2})))) + # colour="CropTyp"
                        # guides(fill=guide_legend(title="")) + 
                        facet_grid(~ CropTyp, scales="free") +
                        stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}, size=1) + 
                        stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                                     fun.min=function(z) { quantile(z,0.1) }, 
                                     fun.max=function(z) { quantile(z,0.9) }, 
                                     alpha=0.4) + # fill = ("darkgreen")
                        stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                                     fun.min=function(z) { quantile(z,0.25) }, 
                                     fun.max=function(z) { quantile(z,0.75) }, 
                                     alpha=0.8) + # fill = ("orange")
                         # scale_fill_brewer(palette = "Set1") +
                        scale_colour_manual(values=c("darkgreen", "dodgerblue")) +
                        scale_fill_manual(values=c("darkgreen", "dodgerblue")) +
                        ggtitle("shifted and scaled") +
                        theme(panel.grid.major = element_line(size=0.2),
                              panel.spacing=unit(.5, "cm"),
                              legend.text=element_text(size=axis_label_font_size), # face="bold"
                              legend.title = element_blank(),
                              legend.position = "none", # "bottom",
                              strip.text = element_text(face="bold", size=axis_text_font_size, color="black"),
                              axis.text = element_text(size=axis_text_font_size, color="black"), # face="bold",
                              axis.text.x = element_blank(),
                              axis.ticks = element_line(color = "black", size = .2),
                              axis.title.x = element_text(margin=margin(t=-1, r=0, b=-1, l=0)),
                              axis.title.y = element_text(size=axis_label_font_size,
                                                          margin=margin(t=0, r=10, b=0, l=0)),
                              plot.title = element_text(lineheight=.8, face="bold", size=title_font_size)
                              ) + 
                        theme_bw()

uptake_plots <- ggpubr::ggarrange(plotlist = list(noShift_uptake, shifted_uptake, yaxis_shifted_uptake),
                                   ncol = 1, nrow = 3, common.legend = TRUE, legend="none")

plot_dir <- paste0(data_dir)
ggsave(filename = "uptakes.png",
       plot=uptake_plots, width=8, height=10.5, units = "in", 
       dpi=300, device = "png", path = plot_dir)

ggsave(filename = "uptakes.png",
       plot=uptake_plots, width=8, height=10.5, units = "in", 
       dpi=300, device = "png", path = plot_dir)
#######################################################
#####
#####    Overlay
#####

corn <- yaxis_scaled_shifted_nit %>% 
        filter(CropTyp=="Corn, Field") %>%
        data.table()

potato <- yaxis_scaled_shifted_nit %>% 
          filter(CropTyp=="Potato") %>%
          data.table()

Deirdre_potato = Deirdre 
Deirdre_corn = Deirdre

Deirdre_potato$CropTyp = "Potato"
Deirdre_corn$CropTyp = "Corn, Field"

dierdre_shift = 130
corn_overlay <- ggplot(corn, aes(x=x_axis, y=smooth_window3)) +  # color=factor(CropTyp), fill=factor(CropTyp))
                # labs(x = "", y=expression(gm^{-2})) +
                labs(x = "", y="") +
                stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}, size=1) + 
                stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                             fun.min=function(z) { quantile(z,0.1) }, 
                             fun.max=function(z) { quantile(z,0.9) }, 
                             alpha=0.4) + # fill = ("darkgreen")
                stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                             fun.min=function(z) { quantile(z,0.25) }, 
                             fun.max=function(z) { quantile(z,0.75) }, 
                             alpha=0.8) +
                geom_line(data=Deirdre_corn, 
                          aes(x=Day-dierdre_shift, y=Inorganic_N_g_m2_mean, color=factor(Product_Type)), 
                          size=1.2) + # )

                #scale_colour_manual(values=c("darkgreen", "dodgerblue")) +
                # scale_fill_manual(values=c("darkgreen", "dodgerblue")) +
                ggtitle("shifted and scaled (corn)") +
                theme(panel.grid.major = element_line(size=0.2),
                      panel.spacing=unit(.5, "cm"),
                      legend.text=element_text(size=axis_label_font_size), # face="bold"
                      legend.title = element_blank(),
                      legend.position = "bottom", # "bottom",
                      strip.text = element_text(face="bold", size=axis_text_font_size, color="black"),
                      axis.text = element_text(size=axis_text_font_size, color="black"), # face="bold",
                      axis.text.x = element_blank(),
                      axis.ticks = element_line(color = "black", size = .2),
                      axis.title.x = element_text(margin=margin(t=-1, r=0, b=-1, l=0)),
                      axis.title.y = element_text(size=axis_label_font_size,
                                                  margin=margin(t=0, r=10, b=0, l=0)),
                      plot.title = element_text(lineheight=.8, face="bold", size=title_font_size)
                      )

potato_overlay <- ggplot(potato, aes(x=x_axis, y=smooth_window3)) +  # color=factor(CropTyp), fill=factor(CropTyp))
                  labs(x = "", y="") +
                  stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}, size=1) + 
                  stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                               fun.min=function(z) { quantile(z,0.1) }, 
                               fun.max=function(z) { quantile(z,0.9) }, 
                               alpha=0.4) + # fill = ("darkgreen")
                  stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                               fun.min=function(z) { quantile(z,0.25) }, 
                               fun.max=function(z) { quantile(z,0.75) }, 
                               alpha=0.8) +
                  geom_line(data=Deirdre_potato, 
                            aes(x=Day-dierdre_shift, y=Inorganic_N_g_m2_mean, color=factor(Product_Type)), 
                            size=1.2) + # )
  
                  #scale_colour_manual(values=c("darkgreen", "dodgerblue")) +
                  # scale_fill_manual(values=c("darkgreen", "dodgerblue")) +
                  ggtitle("shifted and scaled (potato)") +
                  theme(panel.grid.major = element_line(size=0.2),
                        panel.spacing=unit(.5, "cm"),
                        legend.text=element_text(size=axis_label_font_size), # face="bold"
                        legend.title = element_blank(),
                        legend.position = "bottom", # "bottom",
                        strip.text = element_text(face="bold", size=axis_text_font_size, color="black"),
                        axis.text = element_text(size=axis_text_font_size, color="black"), # face="bold",
                        # axis.text.x = element_blank(),
                        axis.ticks = element_line(color = "black", size = .2),
                        axis.title.x = element_text(margin=margin(t=-1, r=0, b=-1, l=0)),
                        axis.title.y = element_text(size=axis_label_font_size,
                                                    margin=margin(t=0, r=10, b=0, l=0)),
                        plot.title = element_text(lineheight=.8, face="bold", size=title_font_size)
                        )

overlay_plots <- ggpubr::ggarrange(plotlist = list(corn_overlay, potato_overlay),
                                   ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")

plot_dir <- paste0(data_dir)
ggsave(filename = paste0("potato_corn_overlay_", dierdre_shift, ".png"),
       plot=overlay_plots, width=8, height=10.5, units = "in", 
       dpi=300, device = "png", path = plot_dir)

#######################################################
#####
#####    Overlay (no worm bedding)
#####
Deirdre_no_wormBedding = Deirdre %>% 
                         filter(Product_Type != "Worm bedding") %>%
                         data.table()
Deirdre_potato_no_wormBedding  = Deirdre_no_wormBedding 
Deirdre_corn_no_wormBedding    = Deirdre_no_wormBedding

Deirdre_potato_no_wormBedding$CropTyp = "Potato"
Deirdre_corn_no_wormBedding$CropTyp = "Corn, Field"

corn_overlay_noBed <- ggplot(corn, aes(x=x_axis, y=smooth_window3)) +  # color=factor(CropTyp), fill=factor(CropTyp))
                      labs(x = "", y="") +
                      stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}, size=1) + 
                      stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                                   fun.min=function(z) { quantile(z,0.1) }, 
                                   fun.max=function(z) { quantile(z,0.9) }, 
                                   alpha=0.4) + # fill = ("darkgreen")
                      stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                                   fun.min=function(z) { quantile(z,0.25) }, 
                                   fun.max=function(z) { quantile(z,0.75) }, 
                                   alpha=0.8) +
                      geom_line(data=Deirdre_corn_no_wormBedding, 
                                aes(x=Day-dierdre_shift, y=Inorganic_N_g_m2_mean, color=factor(Product_Type)), 
                                size=1.2) +

                    #scale_colour_manual(values=c("darkgreen", "dodgerblue")) +
                    # scale_fill_manual(values=c("darkgreen", "dodgerblue")) +
                      ggtitle("shifted and scaled (corn)") +
                      theme(panel.grid.major = element_line(size=0.2),
                            panel.spacing=unit(.5, "cm"),
                            legend.text=element_text(size=axis_label_font_size), # face="bold"
                            legend.title = element_blank(),
                            legend.position = "bottom", # "bottom",
                            strip.text = element_text(face="bold", size=axis_text_font_size, color="black"),
                            axis.text = element_text(size=axis_text_font_size, color="black"), # face="bold",
                            axis.text.x = element_blank(),
                            axis.ticks = element_line(color = "black", size = .2),
                            axis.title.x = element_text(margin=margin(t=-1, r=0, b=-1, l=0)),
                            axis.title.y = element_text(size=axis_label_font_size,
                                                        margin=margin(t=0, r=10, b=0, l=0)),
                            plot.title = element_text(lineheight=.8, face="bold", size=title_font_size)
                            )

potato_overlay_noBed <- ggplot(potato, aes(x=x_axis, y=smooth_window3)) +  # color=factor(CropTyp), fill=factor(CropTyp))
                        labs(x = "", y="") +
                        stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}, size=1) + 
                        stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                                     fun.min=function(z) { quantile(z,0.1) }, 
                                     fun.max=function(z) { quantile(z,0.9) }, 
                                     alpha=0.4) + # fill = ("darkgreen")
                        stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                                     fun.min=function(z) { quantile(z,0.25) }, 
                                     fun.max=function(z) { quantile(z,0.75) }, 
                                     alpha=0.8) +
                        geom_line(data=Deirdre_potato_no_wormBedding, 
                                  aes(x=Day-dierdre_shift, y=Inorganic_N_g_m2_mean, color=factor(Product_Type)), 
                                  size=1.2) +
        
                        #scale_colour_manual(values=c("darkgreen", "dodgerblue")) +
                        # scale_fill_manual(values=c("darkgreen", "dodgerblue")) +
                        ggtitle("shifted and scaled (potato)") +
                        theme(panel.grid.major = element_line(size=0.2),
                              panel.spacing=unit(.5, "cm"),
                              legend.text=element_text(size=axis_label_font_size), # face="bold"
                              legend.title = element_blank(),
                              legend.position = "bottom", # "bottom",
                              strip.text = element_text(face="bold", size=axis_text_font_size, color="black"),
                              axis.text = element_text(size=axis_text_font_size, color="black"), # face="bold",
                              # axis.text.x = element_blank(),
                              axis.ticks = element_line(color = "black", size = .2),
                              axis.title.x = element_text(margin=margin(t=-1, r=0, b=-1, l=0)),
                              axis.title.y = element_text(size=axis_label_font_size,
                                                          margin=margin(t=0, r=10, b=0, l=0)),
                              plot.title = element_text(lineheight=.8, face="bold", size=title_font_size)
                              ) 

overlay_plots_noBed <- ggpubr::ggarrange(plotlist = list(corn_overlay_noBed, potato_overlay_noBed),
                                         ncol=1, nrow=2, common.legend=TRUE, legend="bottom")

plot_dir <- paste0(data_dir)
ggsave(filename = paste0("potato_corn_overlay_noBed", dierdre_shift, ".png"),
       plot=overlay_plots_noBed, width=8, height=10.5, units = "in", 
       dpi=300, device = "png", path = plot_dir)



# In the following plots there are three rows. The first row is observed 
# data through satellite for nitrogen uptake. In the second and third row, 
# we have scaled the data to be between 0 and 1. The difference between 
# second and third row is that in the second row we assumed periodic 
# boundary condition. Think of Cycles in nature; moon around earth? 
# It repeats itself. One may argue that does not happen in a field 
# and a farmer’s practice can change from one year to another. But 
# here we are looking only at one years of data. So, just an idea. 
# You may reject it.
# In the third row, similar to the plots above, the origin of time 
# is set to the DoY at which nitrogen uptake reaches its maximum. 
# (We can reject this idea as well, similar to the argument above 
#     for periodic boundary condition.) 


# Uptakes
