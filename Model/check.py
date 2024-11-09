import os



if __name__ == '__main__':
    epoch = 100
    folderList = os.listdir('./Zhihui_new/')
    folderList.sort()
    folderSelectList = []
    progressList = []
    for folder in folderList:
        if not os.path.exists('./Zhihui_new/'+folder+'/test_'+str(epoch-1)+'.mat'):
            folderSelectList.append(folder)
            progress = epoch
            while(not os.path.exists('./Zhihui_new/'+folder+'/test_'+str(progress-1)+'.mat'))and(progress>0):
                progress -= 1
            progressList.append(progress)
    
    for folderSelect, progress in zip(folderSelectList, progressList):
        print(folderSelect, progress)
    print(len(folderSelectList))