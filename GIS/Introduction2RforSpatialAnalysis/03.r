##
## Code for Chapter 3
##
## Handling Spatial data in R  
## 

## 3.1 OVERVIEW

## 3.2 INTRODUCTION: GISTools

## 3.2.1 Installing and Loading GISTools

library(GISTools)

help(GISTools)
?GISTools

## 3.2.2 Spatial Data in GISTools

data(newhaven)
ls()

plot(roads)

head(data.frame(blocks))
plot(blocks)
plot(roads)

par(mar = c(0,0,0,0)) 
plot(blocks)

plot(blocks)

plot(blocks, lwd = 0.5, border = "grey50") 
plot(breach, col = "red", pch = 1, add = TRUE)

colors()

## 3.2.3 Embellishing the Map

map.scale(534750,152000,miles2ft(2),"Miles",4,0.5)
north.arrow(534750,154000,miles2ft(0.25),col= "lightblue")
title('New Haven, CT.' )

## 3.2.4 Saving Your Map

# load package and data
north.arrow(530650,154000,miles2ft(0.25),col= 'lightblue') 
title('New Haven, CT')

source("newhavenmap.R")

plot(roads,add=TRUE,col= 'blue')

plot(blocks, lwd=3)

plot(roads,add=TRUE,col= "red",lwd=2)

pdf(file= 'map.pdf')
dev.off()

png(file= 'map.png')
dev.off()

## 3.3 MAPPING SPATIAL OBJECTS

## 3.3.1 Introduction

## 3.3.2 Data

rm(list=ls())

library(GISTools)

plot(georgia, col = "red", bg = "wheat")

# do a merge
# plot the spatial layers
	border = "blue")
	col.sub = "blue")

# set some plot parameters
title("georgia")
title("georgia2")

data.frame(georgia)[,13]

# assign some coordinates

# the county indices below were extracted from the data. frame

pl <- pointLabel(Lon[county.tmp], Lat[county.tmp],

plot(georgia, border = "grey", lwd = 0.5) 
plot(georgia.sub, add = TRUE, col = "lightblue") 
plot(georgia.outline, lwd = 2, add = TRUE) 
title("Georgia with a subset of counties")

## 3.3.4 Adding Context

install.packages(c("OpenStreetMap"),depend=T)

# define upper left, lower right corners
lr <- as.vector(cbind(bbox(georgia.sub)[2,1], bbox(georgia.sub)[1,2]))
plot(spTransform(georgia.sub, osm()), add = TRUE, lwd = 2)

install.packages(c("RgoogleMaps"), depend=T)
# load the package

# convert the subset
# download map data and store it
# now plot the layer and the backdrop
# reset the plot margins

## 3.4 MAPPING SPATIAL DATA ATTRIBUTES

## 3.4.1 Introduction

## 3.4.2 Attributes and Data Frames

# load & list the data
summary(blocks)

data.frame(blocks)
head(data.frame(blocks))
colnames(data.frame(blocks))
data.frame(blocks)$P_VACANT
blocks$P_VACANT
attach(data.frame(blocks))
hist(P_VACANT)
detach(data.frame(blocks))

# use kde.points to create a kernel density surface
summary(breach.dens)
head(data.frame(breach.dens))

# use 'as' to coerce this to a SpatialGridDataFrame
summary(breach.dens.grid)

choropleth(blocks, blocks$P_VACANT)
vacant.shades = auto.shading(blocks$P_VACANT)
vacant.shades
choro.legend(533000,161000,vacant.shades)

# set the shading
choropleth(blocks,blocks$P_VACANT,shading=vacant.shades) 
choro.legend(533000,161000,vacant.shades)

display.brewer.all()
brewer.pal(5,'Blues')

vacant.shades = auto.shading(blocks$P_VACANT, cols=brewer.pal(5,"Greens"))
choro.legend(533000,161000,vacant.shades)

vacant.shades = auto.shading(blocks$P_VACANT, n=5, cols=brewer.pal(5,"Blues"), cutter=rangeCuts)

choropleth
auto.shading

## 3.4.4 Mapping Points and Attributes

plot(breach)
plot(blocks) 
plot(breach, add=TRUE)

plot(blocks)

plot(blocks)

plot(blocks)

# examine the Brewer "Reds" colour palette
add.alpha(brewer.pal(5, "Reds"),.50)

par(mar= c(0,0,0,0))
plot(blocks, lwd = 0.7, border = "grey40") 
plot(breach,add=TRUE, pch=1, col= "#DE2D2680")

# load the data
head(quakes)

# define the coordinates
par(mar = c(0,0,0,0))

par(mfrow=c(1,2))
plot(quakes.spdf)
plot(quakes.spdf, pch = 1, col = '#FB6A4A80') 
# reset par(mfrow)

##Not Run##
# install.packages("maps", dep = T)

help("SpatialPolygons-class")

data(georgia)
t2 <- Polygon(tmp[2]); t2 <- Polygons(list(t2), "2") 
t3 <- Polygon(tmp[3]); t3 <- Polygons(list(t3), "3") 
t4 <- Polygon(tmp[4]); t4 <- Polygons(list(t4), "4") 
# create a SpatialPolygons object
plot(tmp.Sp, col = 2:5)
# data.frame(tmp.spdf)

# set some plot parameters
## 1. Plot using choropleth 
choropleth(quakes.spdf, quakes$mag)
## 2. Plot with a different shading scheme & pch
## 3. Plot with a transparency
choropleth(quakes.spdf, quakes$mag, shading = shades, pch = 20)
tmp <- quakes$mag # assign magnitude to tmp
plot(quakes.spdf, cex = tmp*3, pch = 1, col = '#FB6A4A80')

# Set the plot parameters
# reset par(mfrow)

## Info Box
data <- c(3, 6, 9, 99, 54, 32, -102) 
index <- (data == 32 | data <= 6) 
data[index]

library(RgoogleMaps)
PlotOnStaticMap(MyMap,Lat,Long,cex=tmp+0.3,pch=1,

MyMap <- MapBackground(lat=Lat, lon=Long, zoom = 10, maptype = "satellite")
PlotOnStaticMap(MyMap,Lat,Long,cex=tmp+0.3,pch=1,

## 3.4.5 Mapping Lines and Attributes

data(newhaven)
yy = as.vector(c(ymin, ymax, ymax, ymin, ymin)) 
# 2. create a spatial polygon from this
Pl <- Polygon(crds)
tmp <- as.numeric(gsub("clip", "", names(roads.tmp))) 
tmp <- data.frame(roads)[tmp,]

par(mfrow=c(1,3)) # set plot order 
par(mar = c(0,0,0,0)) # set margins
tmp <- roads.tmp$AV_LEGEND
plot(roads.tmp, col = shades[index], lwd = 3)
plot(roads.tmp, lwd = roads.tmp$LENGTH_MI * 10)

## 3.4.6 Mapping Raster Attributes

data(meuse.grid)

plot(meuse.grid$x, meuse.grid$y, asp = 1)

meuse.grid = SpatialPixelsDataFrame(points =

par(mfrow=c(1,2)) # set plot order
# map the dist attribute using the image function 
image(meuse.grid, "dist", col = rainbow(7)) 
image(meuse.grid, "dist", col = heat.colors(7))

# using spplot from the sp package

## 3.5 SIMPLE DESCRIPTIVE STATISTICAL ANALYSES

## 3.5.1 Histograms and Boxplots

data(newhaven)

index <- blocks$P_VACANT > 10 
high.vac <- blocks[index,] 
low.vac <- blocks[!index,]

# set plot parameters and shades
par(mfrow = c(1,2))
attach(data.frame(high.vac))
	names=c("OwnerOcc", "White", "Black"),
attach(data.frame(low.vac))
	names=c("OwnerOcc","White", "Black"), 
	col=cols, cex.axis = 0.7, main = "Low Vacancy")
detach(data.frame(low.vac)) 
# reset par(mfrow) 
par(mfrow=c(1,1))
par(mar=c(5,4,4,2))

## 3.5.2 Scatter Plots and Regressions

plot(blocks$P_VACANT/100, blocks$P_WHITE/100) 
plot(blocks$P_VACANT/100, blocks$P_BLACK/100)

# assign some variables
p.w <- blocks$P_WHITE/100 
p.b <- blocks$P_BLACK/100
mod.2 <- lm(p.vac ~ p.b)


# not run below

# define a factor for the jitter function
# this is to help show densities
plot(jitter(p.vac, fac), jitter(p.w, fac),
	ylab = "Proprtion White / Black",
	col = cols[1], xlim = c(0, 0.8))
# fit some trend lines from the 2 regression model coefficients
legend(0.71, 0.095, legend = "White", bty = "n", cex = 0.8)

## 3.5.3 Mosaic Plots

# populations of each group in each census block
pops <- as.matrix(pops/100)
# the crosstabulations

# mosaic plot
	ylab= 'Vacant Properties > 10 percent', 
	main=ttext,shade=TRUE,las=3,cex=0.8)

## 3.6 SELF-TEST QUESTIONS

## Self-Test Question 1.
# Hints
data(georgia) 	# to load the Georgia data

## Self-Test Question 2. 
# Hints
?par # the help for plot parameters 
par(mfrow = c(1,2)) # set the plot order to be 1 row & 2 columns
	quartz(w=10,h=8) } else {
# Tools
data(newhaven) # to load the New Haven data

## Self-Test Question 3. 
# Hints
# rect() # to draw a rectangle for a legend
help("!") # to examine logic operators
library(GISTools) # for the mapping tools 
data(georgia) # use georgia2 as it has a geographic projection
library(rgeos) # you may need to use install.packages()

## Self-Test Question 4. 
library(GISTools)

summary(quakes.spdf)
proj4string(quakes.spdf) <- CRS("+proj=longlat +ellps=WGS84")

# Hints
?PlotOnStaticMap # for the points
?rgb
# Tools
library(GISTools) # for the mapping tools
library(rgdal) # this has the spatial reference tools
library(RgoogleMaps)
library(PBSmapping)

## ANSWERS TO SELF-TEST QUESTIONS

## Q1.
# load the data and the package
tiff("Quest1.tiff", width=7,height=7,units='in',res=300) 
# define the shading scheme
# add the legend & keys
# close the file

## Q2.
# Code
hist(HSE_UNITS, breaks = 20)
# but with some large outliers
quantileCuts(HSE_UNITS, 5)
sdCuts(HSE_UNITS, 5)
choropleth(blocks,HSE_UNITS,shading=shades) 
choro.legend(533000,161000,shades) 
title("Quantile Cuts", cex.main = 2)
	n = 5, cols = brewer.pal(5, "RdYlGn"))
choro.legend(533000,161000,shades)
shades <- auto.shading(HSE_UNITS, cutter = sdCuts,
choropleth(blocks,HSE_UNITS,shading=shades) 
choro.legend(533000,161000,shades)
title("St. Dev. Cuts", cex.main = 2) 
# 3. Finally detach the data frame 
detach(data.frame(blocks))

## Q3. 
# attach the data frame
rect(850000, 925000, 970000, 966000, col = "white") 
legend(850000, 955000, legend = "Rural",
	bty = "n", pch = 19, col = "chartreuse4") 
legend(850000, 975000, legend = "Not Rural",

## Q4.
library(GISTools) # for the mapping tools 
library(rgdal) # this has the spatial reference tools 
library(RgoogleMaps)
data(newhaven)
Long <- coords[,1]
# convert polys to PBS format
PlotPolysOnStaticMap(MyMap, shp, lwd=0.7,
PlotOnStaticMap(MyMap,Lat,Long,pch=1,col='red', add = TRUE)




