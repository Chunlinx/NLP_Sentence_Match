import os
# configure the project rootDir
def get_project_rootDir():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    print( "Project Directory: {}".format( project_dir ) )
    return project_dir

# if __name__=='__main__':
#     project_rootDir = get_project_rootDir() 
#     # path = os.path.join( project_rootDir, "test/test.txt" )
#     # print( path )