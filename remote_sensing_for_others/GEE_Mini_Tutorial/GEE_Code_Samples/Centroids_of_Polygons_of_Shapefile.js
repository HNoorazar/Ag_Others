/**** Start of imports. If edited, may not auto-convert in the playground. ****/
var four_polygons_wCentroids = ee.FeatureCollection("users/hnoorazar/NASA_trend/four_polygons_wCentroids");
/***** End of imports. If edited, may not auto-convert in the playground. *****/
var SF = ee.FeatureCollection(four_polygons_wCentroids);


function getCentroid(feature) {
  // Keep this list of properties.
  var keepProperties = ['ID', 'county', 'CropTyp', 'Irrigtn', 'LstSrvD'];
  
  // Get the centroid of the feature's geometry.
  var centroid = feature.geometry().centroid();
  
  // Return a new Feature, copying properties from the old Feature.
  return ee.Feature(centroid).copyProperties(feature, keepProperties);
}


var centroids_from_GEE = SF.map(getCentroid);
print ("centroids_from_GEE", centroids_from_GEE);

// Display the results.
Map.centerObject(ee.Geometry.Point([-119.796, 47.05]), 11);
Map.addLayer(centroids_from_GEE, {color: 'red'}, 'GEE centroids');
Map.addLayer(SF, {color: 'blue'}, 'CSV centroids');
