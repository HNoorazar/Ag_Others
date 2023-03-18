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
smoothed_daily     = read.csv(paste0(data_dir, "01_corn_potatoEq2_smoothed_daily.csv"))
# in the following line, the boundary/cycle condition is applied.
# scaled_shifted_nit = read.csv(paste0(data_dir, "03_scaled_shifted_corn_potatoEq2_smoothed.csv"))

# In the following table, each time series is
# centered around its maximum and that is when x-axis is zero.
yaxis_scaled_shifted_nit = read.csv(paste0(data_dir, "03_scaled_yAxis_corn_potatoEq2_smoothed.csv"))

axis_label_font_size <- 12
axis_text_font_size <- axis_label_font_size-2
title_font_size <- axis_label_font_size+2
################################
########
########   Plot shifted uptake
########
################################

# head(scaled_shifted_nit, 2)
# doys <- rep(1:365, 878555/365)
# scaled_shifted_nit$day = doys

doys <- rep(1:365, 878555/365)
smoothed_daily$day = doys

 # stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                  #              fun.min=function(z) { quantile(z,0) }, 
                  #              fun.max=function(z) { quantile(z,1) }, 
                  #              alpha=0.2) +

noShift_uptake <- ggplot(smoothed_daily, aes(x=day, y=smooth_window3, color=factor(CropTyp), fill=factor(CropTyp))) + 
                  labs(x = "", y=expression(paste("nitrogen content ", (gm^{-2})))) +
                  # xlab("") + 
                  # ylab(expression(paste("nitrogen content ", (gm^{-2})))) + 
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
                  scale_colour_manual(values=c("darkgreen", "dodgerblue")) +
                  scale_fill_manual(values=c("darkgreen", "dodgerblue")) +
                  ggtitle("smoothed uptake") +
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
                        labs(x = "", y=expression(paste("nitrogen content ", (gm^{-2})))) +  # colour="CropTyp"
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
                        ggtitle("smoothed, shifted and scaled") +
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


plot_dir <- paste0(data_dir)
ggsave(filename = "noShift_uptake.png",
       plot=noShift_uptake, width=8, height=3, units = "in", 
       dpi=300, device = "png", path = plot_dir)

ggsave(filename = "yaxis_shifted_uptake.png",
       plot=yaxis_shifted_uptake, width=8, height=3, units = "in", 
       dpi=300, device = "png", path = plot_dir)



##################################################
###
###   Find Range in peak time
###
A <- ggplot(smoothed_daily, aes(x=day, y=smooth_window3, color=factor(CropTyp), fill=factor(CropTyp))) + 
                  labs(x = "", y="Nitrogen Uptake") +
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
                  scale_colour_manual(values=c("darkgreen", "dodgerblue")) +
                  scale_fill_manual(values=c("darkgreen", "dodgerblue")) +
                  ggtitle("smoothed uptake") +
                  theme(panel.grid.major = element_line(size=0.2),
                        panel.spacing=unit(.5, "cm"),
                        legend.text=element_text(size=axis_label_font_size), # face="bold"
                        legend.title = element_blank(),
                        legend.position = "none", # "bottom",
                        strip.text = element_text(face="bold", size=axis_text_font_size, color="black"),
                        axis.text = element_text(size=axis_text_font_size, color="black"), # face="bold",
                        # axis.text.x = element_blank(),
                        axis.ticks = element_line(color = "black", size = .2),
                        axis.title.x = element_text(margin=margin(t=-1, r=0, b=-1, l=0)),
                        axis.title.y = element_text(size=axis_label_font_size,
                                                    margin=margin(t=0, r=10, b=0, l=0)),
                        plot.title = element_text(lineheight=.8, face="bold", size=title_font_size)
                        )+
                  geom_vline(xintercept = 225) + 
                  geom_vline(xintercept = 200, linetype="dotted", color = "blue", size=1.5)

plot_dir <- paste0(data_dir)
ggsave(filename = "peak_time_detection.png",
       plot=A, width=8, height=3, units = "in", 
       dpi=300, device = "png", path = plot_dir)


corn_subset <- smoothed_daily %>% 
               filter(CropTyp=="Corn, Field")%>%
               filter(day==225)%>% 
               data.table()

potato_subset <- smoothed_daily %>% 
                 filter(CropTyp=="Potato")%>%
                 filter(day==200)%>% 
                 data.table()

ggplot(corn_subset, aes(x=day, y=smooth_window3, color=factor(CropTyp), fill=factor(CropTyp))) + 
                  labs(x = "", y="Nitrogen Uptake") +
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
                  scale_colour_manual(values=c("darkgreen", "dodgerblue")) +
                  scale_fill_manual(values=c("darkgreen", "dodgerblue")) +
                  ggtitle("smoothed uptake") +
                  theme(panel.grid.major = element_line(size=0.2),
                        panel.spacing=unit(.5, "cm"),
                        legend.text=element_text(size=axis_label_font_size), # face="bold"
                        legend.title = element_blank(),
                        legend.position = "none", # "bottom",
                        strip.text = element_text(face="bold", size=axis_text_font_size, color="black"),
                        axis.text = element_text(size=axis_text_font_size, color="black"), # face="bold",
                        # axis.text.x = element_blank(),
                        axis.ticks = element_line(color = "black", size = .2),
                        axis.title.x = element_text(margin=margin(t=-1, r=0, b=-1, l=0)),
                        axis.title.y = element_text(size=axis_label_font_size,
                                                    margin=margin(t=0, r=10, b=0, l=0)),
                        plot.title = element_text(lineheight=.8, face="bold", size=title_font_size)
                        )




ggplot(corn_subset, aes(x=day, y=smooth_window3, color=factor(CropTyp), fill=factor(CropTyp))) + 
                  labs(x = "", y="Nitrogen Uptake") +
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
                  scale_colour_manual(values=c("darkgreen", "dodgerblue")) +
                  scale_fill_manual(values=c("darkgreen", "dodgerblue")) +
                  ggtitle("smoothed uptake") +
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



