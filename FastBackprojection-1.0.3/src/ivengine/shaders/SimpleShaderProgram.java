package ivengine.shaders;

import static org.lwjgl.opengl.GL20.GL_FRAGMENT_SHADER;
import static org.lwjgl.opengl.GL20.GL_LINK_STATUS;
import static org.lwjgl.opengl.GL20.GL_VERTEX_SHADER;
import static org.lwjgl.opengl.GL32.GL_GEOMETRY_SHADER;
import static org.lwjgl.opengl.GL20.glAttachShader;
import static org.lwjgl.opengl.GL20.glDetachShader;
import static org.lwjgl.opengl.GL20.glDeleteShader;
import static org.lwjgl.opengl.GL20.glCompileShader;
import static org.lwjgl.opengl.GL20.glCreateProgram;
import static org.lwjgl.opengl.GL20.glCreateShader;
import static org.lwjgl.opengl.GL20.glGetProgramInfoLog;
import static org.lwjgl.opengl.GL20.glGetShaderInfoLog;
import static org.lwjgl.opengl.GL20.glDeleteProgram;
import static org.lwjgl.opengl.GL20.glLinkProgram;
import static org.lwjgl.opengl.GL20.glShaderSource;
import static org.lwjgl.opengl.GL20.glUseProgram;
import ivengine.Util;

/**
 * This work is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
 * 
 * @author V�ctor Arellano Vicente (Ivelate)
 * @since 7-12-2013 17:20
 *
 * Wrapper class for a OpenGL shader. Makes shaders way less verbose if used this way.
 * It uses a fragment and a vertex shader, nothing less, nothing more. 
 */
public abstract class SimpleShaderProgram
{
	private boolean VERBOSE=false;
	private int programIndex;
	
	public SimpleShaderProgram(String vertexR,String fragmentR,boolean verbose)
	{
		this(vertexR,fragmentR,null,verbose);
	}
	public SimpleShaderProgram(String vertexR,String fragmentR,String geometryR,boolean verbose)
	{
		this.VERBOSE=verbose;
		
		final String FRAGMENT = Util.readFile(fragmentR);
		final String VERTEX = Util.readFile(vertexR);
		
		int vShader = createShader(GL_VERTEX_SHADER,VERTEX);
		int fShader = createShader(GL_FRAGMENT_SHADER,FRAGMENT);
		int gShader=-1;
		
		this.programIndex = glCreateProgram();
		glAttachShader(this.programIndex , vShader);
		glAttachShader(this.programIndex , fShader);

		if(geometryR!=null){
			final String GEOMETRY=Util.readFile(geometryR);
			gShader=createShader(GL_GEOMETRY_SHADER,GEOMETRY);
			glAttachShader(this.programIndex,gShader);
		}
		
		glLinkProgram(this.programIndex );
		
		glDetachShader(this.programIndex , vShader);
		glDetachShader(this.programIndex , fShader);
		glDeleteShader(vShader);
		glDeleteShader(fShader);
		
		if(geometryR!=null)
		{
			glDetachShader(this.programIndex,gShader);
			glDeleteShader(gShader);
		}
		
		if(VERBOSE){
			String status =  glGetProgramInfoLog(this.programIndex , GL_LINK_STATUS); //Verbose
			if(!status.isEmpty()) {
				System.out.println("Shader link error, shader FS="+fragmentR+" VS="+vertexR);
				System.out.println(status);
			}
		}
	}
	public SimpleShaderProgram(String vertexR,String fragmentR)
	{
		this(vertexR,fragmentR,false);
	}
	private int createShader(int type,String cont)
	{
		int shader=glCreateShader(type);
		glShaderSource(shader, cont);
		glCompileShader(shader);
		
		if(VERBOSE){
			String status = glGetShaderInfoLog(shader, 1000);
			if(!status.isEmpty()) {
				System.out.println("Shader compilation error for "+cont);
				System.out.println(status);
			}
		}
		
		return shader;
	}
	public abstract void setupAttributes();
	public void enable()
	{
		glUseProgram(this.programIndex);
	}
	public void disable()
	{
		glUseProgram(0);
	}
	public abstract int getSize();
	public int getID(){ return this.programIndex;}
	protected abstract void dispose();

	public final void fullClean()
	{
		this.disable();
		glDeleteProgram(this.programIndex);
		this.dispose();
	}
}