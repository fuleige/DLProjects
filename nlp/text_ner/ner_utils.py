

def bio_to_bioes(labels):
    """
    将BIO标注格式转换为BIOES格式
    
    转换规则：
    - O -> O
    - B-XXX + I-XXX + ... + I-XXX -> B-XXX + I-XXX + ... + E-XXX
    - B-XXX (单独) -> S-XXX
    - I-XXX (开头) -> S-XXX (修复错误标注)
    
    Args:
        labels: BIO格式的标签列表
    
    Returns:
        BIOES格式的标签列表
    """
    bioes_labels = []
    n = len(labels)
    
    for i in range(n):
        label = labels[i]
        
        if label == 'O':
            bioes_labels.append('O')
        elif label.startswith('B-'):
            entity_type = label[2:]
            # 检查是否是单字符实体
            if i + 1 >= n or not labels[i + 1].startswith(f'I-{entity_type}'):
                bioes_labels.append(f'S-{entity_type}')
            else:
                bioes_labels.append(f'B-{entity_type}')
        elif label.startswith('I-'):
            entity_type = label[2:]
            # 检查是否是实体结尾
            if i + 1 >= n or not labels[i + 1].startswith(f'I-{entity_type}'):
                bioes_labels.append(f'E-{entity_type}')
            else:
                bioes_labels.append(f'I-{entity_type}')
        else:
            # 保持原样（可能是已经是BIOES格式）
            bioes_labels.append(label)
    
    return bioes_labels


def bioes_to_bio(labels):
    """
    将BIOES标注格式转换为BIO格式
    
    转换规则：
    - O -> O
    - S-XXX -> B-XXX
    - B-XXX -> B-XXX
    - I-XXX -> I-XXX
    - E-XXX -> I-XXX
    
    Args:
        labels: BIOES格式的标签列表
    
    Returns:
        BIO格式的标签列表
    """
    bio_labels = []
    
    for label in labels:
        if label == 'O':
            bio_labels.append('O')
        elif label.startswith('S-'):
            entity_type = label[2:]
            bio_labels.append(f'B-{entity_type}')
        elif label.startswith('E-'):
            entity_type = label[2:]
            bio_labels.append(f'I-{entity_type}')
        else:
            # B-XXX 和 I-XXX 保持不变
            bio_labels.append(label)
    
    return bio_labels
