##
## Code for Chapter 2 
##
## This introduces data types and classes  
## 

## 2.1 INTRODUCTION

## 2.2 THE BASIC INGREDIENTS OF R: VARIABLES AND ASSIGNMENT

# examples of simple assignment
x+y
z

# example of vector assignment
tree.heights
tree.heights**2
sum(tree.heights)
mean(tree.heights)
max.height <- max(tree.heights) 
max.height

tree.heights

# examples of character variable assignment
name
# these can be assigned to a vector of character variables

# an example of a logical variable
northern

## Info Box
##### Example Script #####
## these commands will not do anything  
## Load libraries 
library(GISTools)
source("My.functions.R") 
## load some data
my.data <- read.csv(file = "my.data.csv")
cube.root.func(my.data)

## 2.3 DATA TYPES AND DATA CLASSES

## 2.3.1 Data Types in R

character(8)
is.character("8")

numeric(8)
as.numeric(c("1980","-8","Geography")) 
as.numeric(c(FALSE,TRUE))

logical(7)

data <- c(3, 6, 9, 99, 54, 32, -102) 
# a logical test
sum(data[index])

## 2.3.2 Data Classes in R

# defining vectors
is.vector(tmp)
as.vector(tmp)

# defining matrices
matrix(1:6)
matrix(1:6, ncol = 2)
as.matrix(6:3)
is.matrix(as.matrix(6:3))

flow <- matrix(c(2000, 1243, 543, 1243, 212, 545, 654, 168, 109), c(3,3), byrow=TRUE)
rownames(flow) <- c("Leicester", "Liverpool", "Elsewhere") 
# examine the matrix
# and functions exist to summarise
outflows
z <- c(6,7,8)
z

## Info box
?sum
x <- matrix(c(3,6,8,8,6,1,-1,6,7),c(3,3),byrow=TRUE)
apply(x,1,max)
x[,c(TRUE,FALSE,TRUE)]

# a vector assignment
house.type <- factor(c("People Carrier", "Flat",
house.type

income <-factor(c("High", "High", "Low", "Low", "Low", "Medium", "Low", "Medium"), levels=c("Low", "Medium", "High"))

tmp.list <- list("Lex Comber",c(2005, 2009), "Lecturer", matrix(c(6,3,1,2), c(2,2))) 
tmp.list
# elements of the list can be selected
employee <- list(name="Lex Comber", start.year = 2005, position="Professor")
append(tmp.list, list(c(7,6,9,1)))
lapply(tmp.list[[2]], is.numeric)
lapply(tmp.list, length)

employee <- list(name="Lex Comber", start.year = 2005, position="Professor")
class(employee) <- "staff"
print.staff <- function(x) {
	cat("Job Title: ",x$position, "\n")}

print.staff <- function(x) {
print(unclass(employee))

new.staff <- function(name,year,post) {
	result <- list(name=name, start.year=year,
	position=post)
	class(result) <- "staff"
	return(result)}

leics.uni <- vector(mode='list',3)
leics.uni

## 2.3.3 Self-Test Questions

colours <- factor(c("red","blue","red","white", "silver","red","white","silver", "red","red","white","silver","silver"), levels=c("red","blue","white","silver","black"))

## Self-Test Question 1. 
colours[4] <- "orange"

colours <- factor(c("red","blue","red","white", "silver", "red", "white", "silver", "red","red","white","silver","silver"), levels=c("red","blue","white","silver","black"))

colours2 <-c("red","blue","red","white", "silver","red","white","silver", "red","red","white","silver")

## Self-Test Question 2.
car.type <- factor(c("saloon","saloon","hatchback", "saloon","convertible","hatchback","convertible", "saloon", "hatchback","saloon", "saloon", "saloon", "hatchback"), levels=c("saloon","hatchback","convertible"))
table(car.type, colours)
crosstab <- table(car.type,colours)

## Self-Test Question 3.
engine <- ordered(c("1.1litre","1.3litre","1.1litre", "1.3litre","1.6litre","1.3litre","1.6litre", "1.1litre","1.3litre","1.1litre", "1.1litre", "1.3litre","1.3litre"), levels=c("1.1litre","1.3litre","1.6litre"))
engine > "1.1litre"

## Self-Test Question 4. 
dim(crosstab) # Matrix dimensions
apply(crosstab,1,max)
apply(crosstab,2,max)
example <- c(1.4,2.6,1.1,1.5,1.2) 
which.max(example)

## Self-Test Question 5.
 
## Self-Test Question 6. 
levels(engine)
levels(colours)[which.max(crosstab[,1])]
colnames(crosstab)[which.max(crosstab[,1])]
colnames(crosstab)
crosstab[,1]
which.max(crosstab[,1])

# Defines the function
	return(names(x)[which.max(x)])}
which.max.name(example)

## Self-Test Question 7. 

## Self-Test Question 8.

new.sales.data <- function(colours, car.type) {
	xtab <- table(car.type,colours)
	result <- list(colour=apply(xtab,1,which.max.name),
		type=apply(xtab,2,which.max.name),
		total=sum(xtab)) 
	class(result) <- "sales.data"
this.week <- new.sales.data(colours,car.type) 
this.week

## 2.4 PLOTS

## 2.4.1 Basic Plot Tools

x1 <- rnorm(100) 
y1 <- rnorm(100) 
plot(x1,y1)
plot(x1,y1,pch=16, col='red')

x2 <- seq(0,2*pi,len=100)

plot(x2,y2,type='l', col='darkgreen', lwd=3, ylim=c(-1.2,1.2))

y4 <- cos(x2)
lines(x2, y4, lwd=3, lty=2, col='darkblue')

## Info Box
par(mfrow = c(1,2)) 
# reset

x2 <- seq(0,2*pi,len=100)
par(mfrow = c(1,2))
polygon(y2,y4,col='lightgreen')

# you may have done this in Chapter 1
install.packages("GISTools", depend = T)
# if this fails because of versioning eg with R 3.1.2 try
install.packages("GISTools", type = "source") 
library(GISTools)
# library(GISTools)
appling <- georgia.polys[[1]]
# set the plot extent
polygon(appling, density=14, angle=135)

## 2.4.2 Plot Colours

plot(appling, asp=1, type='n', xlab="Easting", ylab="Northing")
polygon(appling, col=rgb(0,0.5,0.7,0.4))

# set the plot extent
# plot the polygon with a transparency factor 
polygon(appling, col=rgb(0,0.5,0.7,0.4))

plot(appling, asp=1, type='n', xlab="Easting", ylab="Northing")
text(1287000,1053000, "Appling County",cex=1.5) 
text(1287000,1049000, "Georgia",col='darkred')

plot(c(-1.5,1.5),c(-1.5,1.5),asp=1, type='n')

# load some grid data
par(mfrow = c(1,2))
image(mat, "dist")
greenpal <- brewer.pal(7,'Greens')

## 2.5 READING, WRITING, LOADING AND SAVING DATA

## 2.5.1 Text Files

# display the first six rows
dim(appling)
colnames(appling) <- c("X", "Y")

write.csv(appling, file = "test.csv")
write.csv(appling, file = "test.csv", row.names = F)
tmp.appling <- read.csv(file = "test.csv")

## 2.5.2 R Data Files

# this will save everything in the workspace
save(list = c("appling", "georgia.polys"), file = "MyData.RData")
load("MyData.RData")

## 2.5.3 Spatial Data Files

data(georgia)
new.georgia <- readShapePoly("georgia.shp")

## ANSWERS TO SELF-TEST QUESTIONS

## Q4
# Undo the colour[4] <- "orange" line used above
colours[engine > "1.1litre"]
table(car.type[engine < "1.6litre"])
table(colours[(engine >= "1.3litre") & (car.type == "hatchback")])

## Q6
apply(crosstab,1,which.max)

## Q7
apply(crosstab,2,which.max.name)

## Q8
most.popular <- list(colour=apply(crosstab,1,which.max.name),
most.popular

## Q9
print.sales.data <- function(x) { 
	cat("Weekly Sales Data:\n") 
	cat("Most popular colour:\n") 
	for (i in 1:length(x$colour)) {
		cat(sprintf("%12s:%12s\n",names(x$colour)[i], 
		x$colour[i]))}
	for (i in 1:length(x$type)) {
		cat(sprintf("%12s:%12s\n",names(x$type)[i],
		x$type[i]))}