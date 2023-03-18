/**** Start of imports. If edited, may not auto-convert in the playground. ****/
var four_centroids_csv = ee.FeatureCollection("users/hnoorazar/NASA_trend/four_centroids_csv"),
    four_polygons_wCentroids = ee.FeatureCollection("users/hnoorazar/NASA_trend/four_polygons_wCentroids");
/***** End of imports. If edited, may not auto-convert in the playground. *****/
var four_centroids = ee.FeatureCollection(four_centroids_csv);
var SF = ee.FeatureCollection(four_polygons_wCentroids);


var getCentroid = function(feature) {
  // Keep this list of properties.
  var keepProperties = ['ID', 'county', 'CropTyp', 'Irrigtn', 'LstSrvD'];
  
  // Get the centroid of the feature's geometry.
  var centroid = feature.geometry().centroid();
  
  // Return a new Feature, copying properties from the old Feature.
  return ee.Feature(centroid).copyProperties(feature, keepProperties);
};



print (SF);
print (four_centroids.toList(four_centroids.size()));

var centroids_from_GEE = SF.map(getCentroid);
print (centroids_from_GEE);