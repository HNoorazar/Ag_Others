
options(digits=9)


annual_TS_3 <- function(d1, y_colname="mean_days_to_maturity", 
                        fil="time_period", y_label="average number of days to reach maturity"){  
  # In the following line, you can break the color between observed and
  # future. HOWEVER, colour should be name of a column without quotation.
  # So, if you out "fil" it will not work. You must do "get(fil)".
  ggplot(data=d1, aes(year, get(y_colname), colour=get(fil)))+ 
  geom_line(size=2) +
  labs(x="year", y=y_label) + #
  scale_colour_manual(values=c("deepskyblue", "darkgreen")) +
  scale_fill_manual(values=c("deepskyblue", "darkgreen")) +
  facet_wrap(. ~ CRD ~ startDoY ) + 
  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "none",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        # axis.text.x = element_text(angle=20, hjust = 1),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        )

}


annual_TS_timePeriod_groupColor <- function(d1, y_colname="mean_days_to_maturity", 
                                            fil="time_period", y_label="average number of days to reach maturity"){
  ggplot(d1, aes(x=year, y=get(y_colname), colour=startDoY)) + # colour=fil
  geom_line(size=2) +
  facet_wrap(. ~ CRD) +
  labs(x="year", y =y_label, fill = fil) + #
  # guides(fill=guide_legend(title="Time period")) +
  scale_color_manual(values=c("deepskyblue", "darkgreen", "orange")) +
  scale_fill_manual(values=c("deepskyblue", "darkgreen", "orange")) +
  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        # axis.text.x = element_text(angle=20, hjust = 1),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        )
}


annual_TS_timePeriod_groupColor_specialOrder <- function(d1, y_colname="mean_days_to_maturity", 
                                                         fil="time_period", 
                                                         y_label="average number of days to reach maturity"){
  ggplot(d1, aes(x=year, y=get(y_colname), colour=startDoY)) + # colour=fil
  geom_line(size=2) +
  facet_wrap(. ~ CRD)+  # ~ startDoY 
  labs(x="year", y=y_label, fill=fil) + #
  # guides(fill=guide_legend(title="Time period")) +
  # scale_color_manual(values=c("deepskyblue", "darkgreen", "orange")) +
  # scale_fill_manual(values=c("deepskyblue", "darkgreen", "orange")) +
  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        # legend.position = "none",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        # axis.text.x = element_text(angle=20, hjust = 1),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        )
}

box_annual_startDoY_x <- function(dt, y_colname, title_, yLab){
  color_ord = c("grey40", "dodgerblue", "olivedrab4", "red", "orange")
  categ_lab = sort(unique(dt$startDoY))

  x_axis_labels<- sort(as.numeric(array(unique(dt$startDoY))))
  x_axis_labels<- x_axis_labels[seq(1,length(x_axis_labels),2)]

  df <- data.frame(dt)
  df <- (df %>% group_by(startDoY, CRD))
  medians <- (df %>% summarise(med = median(get(y_colname))))

  box_width <- 0.35
  the_theme <- theme(plot.title = element_text(size=13, face="bold"),
                     panel.grid.minor = element_blank(),
                     panel.spacing=unit(.5, "cm"),
                     legend.margin=margin(t=.1, r=0, b=.1, l=0, unit='cm'),
                     legend.title = element_blank(),
                     legend.position="none", 
                     legend.key.size = unit(1.5, "line"),
                     legend.spacing.x = unit(.5, 'cm'),
                     panel.grid.major = element_line(size=0.1),
                     axis.ticks = element_line(color="black", size=.2),
                     strip.text = element_text(size=12, face="bold"),
                     legend.text=element_text(size=12),
                     axis.title.x=element_text(size=14, face="plain", margin=margin(t=12, r=0, b=0, l=0)),
                     axis.title.y=element_text(size=14, face="plain"),
                     # axis.text.x = element_blank(), # element_text(size= 12, face = "plain", color="black", angle=-30),
                     axis.text.x = element_text(size= 12, face="plain", color="black", angle=-90),
                     axis.text.y = element_text(size=12, color="black"),
                     axis.ticks.x = element_blank())

  safe_b <- ggplot(data=dt, aes(x=startDoY, y=get(y_colname), fill=startDoY)) +
            geom_boxplot(outlier.size = .15, 
                         notch=F, width=box_width, 
                         lwd=.3, alpha=.8) +
            facet_grid(~ CRD ~ time_period) +
            the_theme + 
            scale_x_discrete(breaks=x_axis_labels,
                             labels=x_axis_labels) +

            # scale_color_manual(values = color_ord, labels = categ_lab,
            #                    name = "Time\nPeriod", limits = color_ord) +
            ggtitle(lab=title_) + 
            labs(x="accumulation start (DoY)", y=yLab) 
            # geom_text(data = medians, 
            #           aes(label=sprintf("%1.0f", medians$med), 
            #               y=medians$med), 
            #               colour = "black", fontface = "bold",
            #               size=5, 
            #               position=position_dodge(.09),
            #               vjust = 1)
  return (safe_b)
}


annual_TS <- function(d1, y_colname="mean_days_to_maturity", fil="maturity age", 
                      y_label_="average number of days to reach maturity"){
  if (y_colname=="mean_days_to_maturity"){
     cls<-"deepskyblue" # bloom color
      } else { # y_colname == "thresh_DoY")
        cls<-"darkgreen" # threshold color
  }
  # d1$chill_season <- gsub("chill_", "", d1$chill_season)
  # d1$chill_season <- substr(d1$chill_season, 1, 4)
  # d1$chill_season <- as.numeric(d1$chill_season)
  
  # xbreaks <- sort(unique(d1$chill_season))
  # xbreaks <- xbreaks[seq(1, length(xbreaks), 15)]

  ggplot(d1, aes(x=year, y=get(y_colname), fill=fil, group=STASD_N)) +
  geom_line(size=2, color=cls)+
  labs(x="year", y =y_label_) + #, fill = "Climate Group"
  # guides(fill=guide_legend(title="Time period")) +
  facet_wrap(. ~ startDoY ~ CRD) + # scales = "free"
  
  # scale_x_continuous(breaks=seq(1970, 2100, 10)) +
  # scale_x_continuous(breaks = xbreaks) +
  # scale_y_continuous(breaks = chill_doy_map$day_count_since_sept, labels = chill_doy_map$letter_day) +
  scale_color_manual(values=cls) +
  scale_fill_manual(values=cls) +
  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        # axis.text.x = element_text(angle=20, hjust = 1),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        ) 
}




