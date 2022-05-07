Plotin3d <-
function(df, order, colsc, compliancecol=NWidthDifference, showwrong=F){  #ex:Plotin3d(Day7, orderwcs, colors5, NWidthDifference)  df=dataframe, order=order you want them in, colsc=color scheme (vector with 27 colors), compliancecol = what column you want plotted
    library("rgl", lib.loc="~/R/win-library/3.4")
    compliancecol <- deparse(substitute(compliancecol))
    df$color<-as.character(df$Object)
    df$color2<-as.character(df$Response)#take out later
    for (i in 1:27){
      df$color[df$color==order[i]] <- colsc[i]
      df$color2[df$color2==order[i]] <- colsc[i]
    }
    open3d()
    par3d(windowRect = c(0, 25, 1922, 1040)) #change window size and location
    plot3d(x = df$NForce, y = df$NSWidth, z = df[[compliancecol]],
           xlab = "", ylab = "", zlab = "",
           size = 2, type = "s", col = df$color) #change point size and type
   if (showwrong==T) {
     plot3d(add = T, x = df$NForce[df$CGuess==F], y = df$NSWidth[df$CGuess==F], z = df[[compliancecol]][df$CGuess==F],
           xlab = "", ylab = "", zlab = "",
           size = 2.5, type = "s", col = rgb(.5, .5, .5), alpha = .5)#col = rgb(.5, .5, .5) or df$color2
   }
    par3d(cex=1.2)
    title3d(xlab = "Force", ylab = "Size", zlab = "Compliance", cex=1.2)
    legend3d("topright", legend = paste('Object', order), pch = 16,
             col = colsc, cex=2.2, inset=c(0.02), pt.cex = 5) #change legend characteristics
}
