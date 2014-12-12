/**
 * License: GPL
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package org.janelia.alignment;

import java.io.FileReader;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Writer;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.lang.StringBuilder;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Type;

import mpicbg.models.PointMatch;
import mpicbg.models.AbstractModel;
import mpicbg.models.TranslationModel2D;
import mpicbg.models.AffineModel2D;
import mpicbg.models.SimilarityModel2D;
import mpicbg.models.RigidModel2D;
import mpicbg.models.HomographyModel2D;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.imagefeatures.Feature;
import mpicbg.ij.FeatureTransform;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;
import com.google.gson.reflect.TypeToken;
import com.thoughtworks.xstream.XStream;
import com.thoughtworks.xstream.io.xml.DomDriver;
import mpicbg.imagefeatures.Feature;


/**
 * 
 * @author Seymour Knowles-Barley,edited by Forrest Collman
 */
public class MatchSiftFeaturesFromFile
{
	@Parameters
	static private class Params
	{
	  
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--featurefile1", description = "First feature file", required = true )
        private String featurefile1;
        
        @Parameter( names = "--featurefile2", description = "Second feature file", required = true )
        private String featurefile2;
        
        @Parameter( names = "--outputTransformFile", description = "Path to output transform xstream object", required = false )
        public String outputTransformFile = null;
        
        @Parameter( names = "--outputInliersFile", description = "Path to output xstream inliers", required = false )
        public String outputInliersFile = null;
        
        @Parameter( names = "--modelType", description = "type of model (0=translation,1=rigid,2=similarity,3=affine", required = false )
        public int modelType = 1;
        //0=translation,1=rigid,2=similarity,3=affine
        
        @Parameter( names = "--minNumInliers", description = "Minimum number of inliers to output a model/inliers", required = false )
        public int minNumInliers = 20;
        
        @Parameter( names = "--maxEpsilon", description = "Maximum distance to consider a point an inlier after fitting transform", required = false )
        public float maxEpsilon = 1.5f;
        
        @Parameter( names = "--Niters", description = "max number of iterations for ransac", required = false )
        public int Niters = 1000;
        
        @Parameter( names = "--minInlierRatio", description = "minimum ratio of inliers/total features to consider it a fit", required = false )
        public float minInlierRatio = 0.0f;
        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
        @Parameter( names = "--rod", description = "ROD", required = false )
        public float rod = 0.95f;
	}
	
	private MatchSiftFeaturesFromFile() {}
	
	public static Object loadObjectFromFile(String path) {

    try{
      //filename is filepath string
      BufferedReader br = new BufferedReader(new FileReader(new File(path)));
      String line;
      StringBuilder sb = new StringBuilder();
      
      while((line=br.readLine())!= null){
          sb.append(line);
      }
      String xml = sb.toString();
		  br.close();
		  
	  	XStream xstream = new XStream();
		  
		  return xstream.fromXML(xml);
    }
      catch (final FileNotFoundException e){
        e.printStackTrace();
        return null;
      }
      catch (final IOException e ){
        e.printStackTrace();
        return null;
      }

}
	public static boolean writeObjectToFile(Object o,String path) {

      //filename is filepath string
       try {
            File file = new File(path);
            file.getParentFile().mkdirs();
            Writer writer = new FileWriter(file);
            //Gson gson = new GsonBuilder().create();
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(o, writer);
            writer.close();
        }
        catch ( final IOException e )
        {
            System.err.println( "Error writing JSON file: " + path );
            e.printStackTrace( System.err );
        } 
    
        catch (final Exception e){
            e.printStackTrace();
            return false;
        }
        return true;

}

	public static void main( final String[] args )
	{
		
		final Params params = new Params();
		try
        {
			final JCommander jc = new JCommander( params, args );
        	if ( params.help )
            {
        		jc.usage();
                return;
            }
        }
        catch ( final Exception e )
        {
        	e.printStackTrace();
            final JCommander jc = new JCommander( params );
        	jc.setProgramName( "java [-options] -cp render.jar org.janelia.alignment.MatchSiftFeaturesXstream" );
        	jc.usage(); 
        	return;
        }

    

		/* open featurespec */
		System.out.println("reading features..");
	  Type listType = new TypeToken<ArrayList<Feature>>() {          }.getType();
		final List<Feature> fs1;
		final List<Feature> fs2;
		try
		{
		  final Gson gson = new Gson();
		  fs1 = gson.fromJson( new FileReader( params.featurefile1 ), listType);
			fs2 = gson.fromJson( new FileReader( params.featurefile2 ), listType );
			
			//fs1 = (List<Feature>) loadObjectFromFile( params.featurefile1);
			//fs2 = (List<Feature>) loadObjectFromFile( params.featurefile2);
		}
		catch ( final Exception e )
		{
			e.printStackTrace( System.err );
			return;
		}

   
    
    System.out.println("matching features..");
		final List< PointMatch > candidates = new ArrayList< PointMatch >();
		FeatureTransform.matchFeatures( fs1, fs2, candidates, params.rod );

		ArrayList<PointMatch> inliers = new ArrayList< PointMatch >();
    AbstractModel< ? > model;
		switch ( params.modelType )
		{
		case 0:
			model = new TranslationModel2D();
			break;
		case 1:
			model = new RigidModel2D();
			break;
		case 2:
			model = new SimilarityModel2D();
			break;
		case 3:
			model = new AffineModel2D();
			break;
		case 4:
			model = new HomographyModel2D();
			break;
		default:
			return;
		}
		
		System.out.println("fitting model ..");
		boolean modelFound;
		try
		{
			modelFound = model.filterRansac(
					candidates,
					inliers,
					params.Niters,
					params.maxEpsilon,
					params.minInlierRatio,
					params.minNumInliers );
			System.out.println("model found with " + inliers.size() + " inliers");
		}
		catch ( final NotEnoughDataPointsException e )
		{
		  System.out.println("no model found..");
			modelFound = false;
		}
   
    System.out.println("writing output..");
    

   //write fit model to file if the model was found and it is good enough
    if (modelFound) {
      if (params.outputTransformFile != null){
        writeObjectToFile(model,params.outputTransformFile);
      }
      if (params.outputInliersFile != null){
        writeObjectToFile(inliers,params.outputInliersFile);
      }
    }
	  
	
	}
}
