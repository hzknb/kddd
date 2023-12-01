// 提交表单
function submitFrom() {
    // 验证数据是否输入以及是否合规

    // 验证教师框架
    if (!myForm.frameT.value) {
        alert("信息不完全");
        myForm.frameT.focus();
        return;
    }

    // 验证学生框架
    if (!myForm.frameS.value) {
        alert("信息不完全");
        myForm.frameS.focus();
        return;
    }

    // 验证损失函数
    if (!myForm.loss.value) {
        alert("信息不完全");
        myForm.button2.focus();
        return;
    }
    
    // 验证项目名称
    if (!myForm.project_name.value) {
        alert("信息不完全");
        myForm.name.focus();
        return;
    }

    // 验证温度
    if (!myForm.T.value) {
        alert("信息不完全");
        myForm.T.focus();
        return;
    }
    else if (parseFloat(myForm.T.value) <= 0) {
        alert("参数错误");
        myForm.T.focus();
        return;
    }

    // 验证alpha
    if (!myForm.alpha.value) {
        alert("信息不完全");
        myForm.alpha.focus();
        return;
    }
    else if (parseFloat(myForm.alpha.value) < 0 && parseFloat(myForm.alpha.value) > 1) {
        alert("参数错误");
        myForm.alpha.focus();
        return;
    }

    // 验证epoch
    if (!myForm.epoch.value) {
        alert("信息不完全");
        myForm.epoch.focus();
        return;
    }
    else if (parseFloat(myForm.epoch.value) <= 0) {
        alert("参数错误"); 
        myForm.epoch.focus();
        return;
    }

    // 验证教师模型
    if (!myForm.module_T.value) {
        alert("信息不完全");
        myForm.module_T.focus();
        return;
    }

    // 验证学生模型
    if (!myForm.module_S.value) {
        alert("信息不完全");
        myForm.module_S.focus();
        return;
    }
    
    // 验证样本集
    if (!myForm.data_sample.value) {
        alert("信息不完全");
        myForm.data.focus();
        return;
    }

    // 设置表单提交格式以及ip端口
    myForm.method = 'POST';
    myForm.action = "http://127.0.0.1:5050/getdata";

    //提交表单
    myForm.submit();
}
