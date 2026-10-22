
from typing import Dict, Optional, List

import os
import re

import numpy as np

from amdata.annotation import Annotation

class AnnotationHTK(Annotation):
    """
    Annotation object to access HTK lab files.

    """

    def __init__(self,
        file_name: str="",
        file_prefix: str="",
        file_suffix: str="",
        param_template: Dict[str,str]=None,
        keys: List[str]=None,
        **kwargs
    ) -> None :
        """
        Args:
            file_name (str, optional): path of the annotation file. Defaults to None.
            file_prefix (str, optional): prefix of the annotation file (if file_name is None). Defaults to None.
            file_suffix (str, optional): suffix of the annotation file (if file_name is None). Defaults to None.
            param_template (Dict[str,str], optional): _description_. Defaults to None.
            keys (List[str], optional): NOT USED
        """
        super().__init__(file_name=file_name,file_prefix=file_prefix,file_suffix=file_suffix,param_template=param_template,**kwargs)

        self._keys = keys
        self._data = None

        if self._file_name:
            self.load(file_name=self._file_name,file_prefix=self._file_prefix,file_suffix=self._file_suffix)


    def load(
        self,
        file_name: str="",
        file_prefix: str="",
        file_suffix: str=""        
    ) -> None :

        if file_name is not None:
            self._file_name = file_name
    
        if file_prefix is not None:
            self._file_prefix = file_prefix

        if file_suffix is not None:
            self._file_suffix = file_suffix
        
        if not self._file_name and self._file_prefix and self._file_suffix:
            self._file_name = os.path.join(self._file_prefix,self._file_suffix)

        if not os.path.exists(self._file_name):
            raise FileNotFoundError(f"File {self._file_name} not found.")

        htk = HTKLabFile(file_name=self._file_name)
        self._data =  htk.get_element_list()


    def save(
        self
    ) -> None :
        print("save not implemented")

    def get_data(
        self,
        start_index: int=None,
        end_index: int=None
    ) -> List[Dict] :
        """
        Get data from the htk file.

        Args:
            start_index (int, optional): start index. Defaults to None.
            end_index (int, optional): end index. Defaults to None.

        Returns:
            List of dictionary with the following elements:
                - start_time (`float`)
                - end_time (`float`)
                - label (`str`)
        """

        if start_index is None:
            start_index =0
        if end_index is None:
            end_index = len(self._data)

        annotation = []
        for i in range(start_index,end_index):
            element = {}
            element['start_time'] = self._data[i].get_time_start()
            element['end_time'] = self._data[i].get_time_end()
            element['label'] = self._data[i].get_label()
            annotation.append(element)

        return annotation

    def __call__(self,
        file_suffix=None,
        **kwargs
    ) -> Optional[Dict]:

        output = None

        if file_suffix is not None:
            file_name = os.path.join(self._file_prefix, file_suffix)
            if file_name != self._file_name:
                self.load(file_name=file_name)

        if self._file_name:
            output = self.get_data(**kwargs)
        
        return output


class HTKLabElement: # annotations csv

    def __init__(self, **args):
        
        self._time_start = 0.0
        self._time_end = 0.0
        self._label = []
        self._score = []
        
    ## @method int get_time_start(self, **args)
    # @brief get the time start at index position in the lab sequence
    # @details
    # @param[in] **args
    #    - index => int, index position in the lab sequence
    # @throw Error::Base if index is not in range
    def get_time_start(self, **args):
        return self._time_start

    def set_time_start(self, **args):
        
        if 'time' not in args:
            raise Exception('argument <time> is missing')
   
        timeValue = float(args['time'])           
        self._time_start = timeValue
        
    def get_time_end(self, **args):
        return self._time_end
    
    def set_time_end(self, **args):
        
        if 'time' not in args:
            raise Exception('argument <time> is missing')
          
        timeValue = float(args['time'])              
        self._time_end = timeValue
        
    
    def get_label_n(self, **args):
        return len(self._label)
    
    def get_label_list(self, **args):
        return self._label
    
    def get_label(self, **args):
    
        index = 0
        if 'index' in args:
            index = int(args['index'])
            
        nLabel = len(self._label)
        
        if ((index<0) or (index>=nLabel)):
            raise Exception("index (%d) is out of range [0,%d)" % (index,nLabel))
        
        return self._label[index]


    def set_label(self, **args):
    
        if 'index' not in args:
            raise Exception('argument <index> is missing')

        if 'label' not in args:
            raise Exception('argument <label> is missing')
    
        index = int(args['index']) 
        nLabel = len(self._label)
        
        if ((index<0) or (index>nLabel)):
            raise Exception("index (%d) is out of range [0,%d]" % (index,nLabel))
        
        if(index<nLabel):
            self._label[index] = args['label']
        else:
            self._label.append(args['label'])
 
 
    def get_score_n(self, **args):
    
        nScore = len(self._score)   
        return nScore
 

    def get_score_list(self, **args):
        return self._score
            
    def get_score(self, **args):
    
        if 'index' not in args:
            raise Exception('argument <index> is missing')
    
        index = int(args['index']) 
        nScore = len(self._score)
        
        if ((index<0) or (index>=nScore)):
            raise Exception("index (%d) is out of range [0,%d)" % (index,nScore))
        
        return self._score[index]


    def set_score(self, **args):
    
        if 'index' not in args:
            raise Exception('argument <index> is missing')

        if 'score' not in args:
            raise Exception('argument <score> is missing')
    
        index = int(args['index']) 
        nScore = len(self._score)
        
        if ((index<0) or (index>nScore)):
            raise Exception("index (%d) is out of range [0,%d]" % (index,nScore))
        
        if(index<nScore):
            self._score[index] = args['score']
        else:
            self._score.append(args['score'])


class HTKLabFile:
    """ Class to load HTK lab file.

        Details on the format can be found online in HTK Book chapter 5.7.1.

        Not everything is imindexplemented 100%, but most features should be supported.
    """

    def __init__(self, **args):
        
        self._element = []
        self._comment = ''
        self._file_name = ''
        
        if 'comment' in args:
            self.set_comment(comment=args['comment'])

        if 'file_name' in args:
            self.set_file_name(file_name=args['file_name'])
            self.load()
        
            
    ## @method int get_comment(%args)
    # @brief get comment
    # @details
    # @param[in]  %args            
    def get_comment(self, **args):
    
        return self._comment
    
    ## @method int set_comment(%args)alignment
    # @brief set a comment
    # @details
    # @param[in]  %args
    #    - comment => strin, comment to insert
    # @throw Error::Base if comment is missing         
    def set_comment(self, **args):
    
        if 'n' not in args:
            raise Exception('argument <comment> is missing')
    
        self._comment = args['comment']
            
                
    ## @method int get_file_name(%args)
    # @brief get file name
    # @details
    # @param[in]  %args            
    def get_file_name(self, **args):
    
        return self._file_name
    
    
    ## @method int set_file_name(%args)
    # @brief set file name
    # @details
    # @param[in]  %args            
    def set_file_name(self, **args):
        
        if 'file_name' not in args:
            raise Exception('argument <file_name> is missing')
        
        self._file_name = args['file_name']
    
    
    ## @method int get_element(self, **args)
    # @brief
    # @details
    # @param[in] **args
    # @throw Exception if index is not in range
    def get_element(self, **args):
    
        if 'index' not in args:
            raise Exception('argument <index> is missing')
    
        index = int(args['index'])
           
        nElement = len(self._element)
        
        if ((index<0) or (index>=nElement)):
            raise Exception("index (%d) is out of range [0,%d)" % (index,nElement))
        
        return self._element[index]
    
    
    ## @method int get_element(self, **args)
    # @brief
    # @detailsget_element_list
    # @param[in] **args
    # @throw Exception if index is not in range
    def get_element_n(self, **args):
          
        return len(self._element)
    
    
    ## @method int get_element(self, **args)
    # @brief
    # @detailsget_element_list
    # @param[in] **args
    # @throw Exception if index is not in range
    def get_element_list(self, **args):
          
        return self._element
    
    
    ## @method int get_element(self, **args)
    # @brief
    # @details
    # @param[in] **args
    # @throw Exception if index is not in range
    def set_element(self, **args):
    
        if 'index' not in args:
            raise Exception('argument <index> is missing')

        if 'element' not in args:
            raise Exception('argument <element> is missing')
    
        index = int(args['index'])
           
        nElement = len(self._element)
        
        if ((index<0) or (index>nElement)):
            raise Exception("index (%d) is out of range [0,%d)" % (index,nElement))

        if(index<nElement):
            self._element[index] = args['element']
        else:
            self._element.append(args['element'])

    ## @method void add(%args)
    # @brief add a new label description at index in the lab sequence containing n items (n>=0)
    # @details
    # @param[in]  %args
    #    - index => int, index position in the lab sequence, if index == n <=> insertion at end
    #    - label => array of strings (multiple labels), mandatory
    #    - score => array of score values (multiple scores), optional
    #    - time_start => number, start time, optional
    #    - time_end => number, end time, optional
    #    - time_base => string, base of time interpretation, 's', '100ns' (default 100 ns)
    # @throw Error::Base if label is missign
    # @throw Error::Base if index        startTime = -1.0
    def add_element(self, **args):
           
        if 'element' not in args:
            raise Exception('argument <element> is missing')     

        element = args['element']      
        
        self.set_element(index=self.get_element_n(),element=element)
    
    ## @method void add(%args)
    # @brief add a new label description at index in the lab sequence containing n items (n>=0)
    # @details
    # @param[in]  %args
    #    - index => int, index position in the lab sequence, if index == n <=> insertion at end
    #    - label => array of strings (multiple labels), mandatory
    #    - score => array of score values (multiple scores), optional
    #    - time_start => number, start time, optional
    #    - time_end => number, end time, optional
    #    - time_base => string, base of time interpretation, 's', '100ns' (default 100 ns)
    # @throw Error::Base if label is missign
    # @throw Error::Base if index        startTime = -1.0
    def add(self, **args):
        
        #if not 'index' in args:
        #    raise Exception('argument <index> is missing')
        
        #index = int(args['index'])
     
        if 'label' not in args:
            raise Exception('argument <label> is missing')     
        labelList = args['label']
        
        scoreList = []
        if 'score' in args:
            scoreList =  args['score']
            
        time_base = self.time_base_htk()
        if 'time_base' in args:
            time_base = args['time_base']
            
            
        startTime = -1.0
        if 'time_start' in args:
            startTime = float(args['time_start'])
     
        endTime = -1.0
        if 'time_end' in args:
            endTime = float(args['time_end'])
        
        
        element = HTKLabElement(time_base=time_base)
        
        element.set_time_start(time=startTime)
        element.set_time_end(time=endTime)
        
        index = 0 
        for label in labelList:
            element.set_label(index=index,label=label)         
            index = index + 1
     
        index = 0 
        for score in scoreList:
            element.set_score(index=index,score=score)            
            index = index + 1

        self.set_element(index=self.get_element_n(),element=element)
     
     
    ## @method void load(%args)
    # @brief 
    # @details
    # @param[in]  %args
    #    - file_name => string, HTKLab file to load
    # @throw Error::Base if file is not defined    
    def save(self, **args):
        
        if 'file_name' in args:
            self.set_file_name(file_name=args['file_name'])

        timeBase = HTKLabFile.time_base_htk()
        if 'time_base' in args:
            if args['time_base'] == HTKLabFile.time_base_second():
                timeBase = HTKLabFile.time_base_second()
            elif args['time_base'] == HTKLabFile.time_base_htk():
                timeBase = HTKLabFile.time_base_htk()
            else:
                raise Exception('argument time_base (%s) is not in range' % (args['time_base']))

        elementList = self.get_element_list()

        fileHandle = open(self._file_name, "w")

        for element in elementList:

            start = element.get_time_start()
            if timeBase==HTKLabFile.time_base_htk():
                start = int(start*10000000.0)

            end = element.get_time_end()
            if timeBase==HTKLabFile.time_base_htk():
                end = int(end*10000000.0)

            labels = element.get_label_list()
            scores =  element.get_score_list()
            nScore = len(scores)

            fileHandle.write("{:10d}\t{:10d}\t".format(start, end))

            for k,label in enumerate(labels):
                if k<nScore:
                    fileHandle.write("{} {} ".format(label,scores[k]))
                else:
                    fileHandle.write("{} ".format(label))

            fileHandle.write("\n")

            #label = ' '.join(element.get_label_list())
            #fileHandle.write("{:10d}\t{:10d}\t{}\n".format(start, end, label))

        fileHandle.close()


    ## @method void load(%args)
    # @brief 
    # @details
    # @param[in]  %args
    #    - file_name => string, HTKLab file to load
    # @throw Error::Base if file is not defined    
    def load(self, **args):
        
        if 'file_name' in args:
            self.set_file_name(file_name=args['file_name'])

        timeBase = HTKLabFile.time_base_htk()
        if 'time_base' in args:
            if args['time_base'] == HTKLabFile.time_base_second():
                timeBase = HTKLabFile.time_base_second()
            elif args['time_base'] == HTKLabFile.time_base_htk():
                timeBase = HTKLabFile.time_base_htk()
            else:
                raise Exception('argument time_base (%s) is not in range' % (args['time_base']))

        fileHandle = open(self._file_name, "r")
        
        indexElement = 0
        
        for line in fileHandle:
            
            line = line.rstrip("\n")
            
            line = re.sub(r'^\s+','',line)
            line = re.sub(r'\s+$','',line)
            
            m=re.match(r'^#(.+)',line)
            
            if m:
                self.set_comment(m.group(1))
            else:
                tokenList = re.split(r'\s+',line)
                nToken = len(tokenList)

                if nToken<2:
                    raise Exception('line <%s>, label is missing' % (line))
                
                if nToken==2:
                    tokenList.append(' ')
                    nToken = len(tokenList)
                    
                labelList = []
                startTime = -1
                endTime = -1
                scoreList = []
                
                indexToken = 0
                
                #  next token does not match an integer/float, second token is a label => end is default (-1)
                if (not re.match(r'\d+$',tokenList[1])) and (not re.match(r'[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',tokenList[1])):
                    labelList.append(tokenList[0])
                    indexToken = indexToken + 1
                else:
                    if (not re.match(r'\d+$',tokenList[1])) and (not re.match(r'[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',tokenList[1])):              
                        startTime = float(tokenList[0])
                        if re.match(r'\.',tokenList[1]):
                            timeBase = 's'
                        else:
                            timeBase = '100ns'   
                        indexToken = indexToken + 2
                    else:
                        #  next token match an integer or a float, third token is a label
                        startTime = float(tokenList[0])
                        if(re.match(r'[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',tokenList[0])):
                            timeBase = 's'
                        else:
                            timeBase = '100ns'
                            
                        endTime = float(tokenList[1])
                        if(re.match(r'[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',tokenList[1])):
                            timeBase = 's'
                        else:
                            timeBase = '100ns'                                                    
                        
                        if nToken<3:
                            raise Exception('line <%s>, label is missing' % (line))                          
                        
                        labelList.append(tokenList[2])
                        indexToken = indexToken + 3
                        
                while(indexToken<nToken):
                    if(re.match(r'[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',tokenList[indexToken])):
                        # score value
                        scoreList.append(tokenList[indexToken])
                    else:
                        # auxiliary label
                        labelList.append(tokenList[indexToken])
                    indexToken = indexToken + 1

                element = HTKLabElement()
      
                if timeBase==HTKLabFile.time_base_htk():
                    startTime = float(startTime)/10000000.0
                    endTime = float(endTime)/10000000.0

                element.set_time_start(time=startTime)
                element.set_time_end(time=endTime)
                  
                index = 0 
                for label in labelList:
                    element.set_label(index=index,label=label)
                    index = index + 1
     
                index = 0
                for score in scoreList:
                    element.set_score(index=index,score=score)            
                    index = index + 1

                self.set_element(index=indexElement,element=element)
                
            indexElement = indexElement + 1
      
        fileHandle.close()
     
     
    ## @cmethod string time_base_second(cls,**args)
    # @brief time base is one second
    # @details
    # @param[in], key/value arguments
    # @return int
    @classmethod
    def time_base_second(cls,**args):
        return 's' 
    
    ## @cmethod string time_base_htk(cls,**args)
    # @brief time base is one second
    # @details
    # @param[in], key/value arguments
    # @return int
    @classmethod
    def time_base_htk(cls,**args):
        return '100ns' 
 

