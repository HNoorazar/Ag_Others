

var xmin = -122.0 ;
var xmax= -120.2;

var ymin = 35.35;
var ymax = 37.2;

var xmed2 = (xmin + xmax) / 2.0;
var ROI = ee.Geometry.Polygon([[xmed2, ymin], [xmed2, ymax], [xmax, ymax], [xmax, ymin], [xmed2, ymin]]);
ROI = [ROI];
ROI = ee.FeatureCollection(ROI);


var data_source = 'LANDSAT/LE07/C02/T1_L2';
var start_date = '2014-01-01';
var end_date = '2014-01-10';

var imageC = ee.ImageCollection(data_source)
               .filterDate(start_date, end_date)
               .filterBounds(ROI)
               .map(function(image){return image.clip(ROI)})
               .sort('system:time_start', true);
               
print (imageC.size());

/*
 The following for-loop will not work since imageC.size() is a server-side object,
 but the for-loop is a client-side shit.

*/

for (var n=0; n < imageC.size(); n++) {
  print ("loop with imageC.size()", n);
}


/*
 The following for-loops will not work.
*/

for (var n=0; n < imageC.size().getInfo(); n++) {
  print ("loop with imageC.size().getInfo()", n);
}


for (var n=0; n < 2; n++) {
  print ("hard-coded", n);
}


/*
 You can do string concatenation in clinet-side but a 
 server-side object cannot be combined with a client side object.
 
 The following example is useful when, for example, you want to add
 system:time_start to the name of the output file:
*/

var x = "A_" + "3";
print ("x", x);


var y = "A_" + imageC.size().getInfo();
print ("y", y);

/*
 Good luck with the following
*/
var z = "A_" + imageC.size();
print ("z", z);



